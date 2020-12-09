import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
import numpy as np
from numpy.random import rand
from numpy.random import shuffle
import easyocr
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_sequences(tokenizer, length, lines):
	tokenized = tokenizer.texts_to_sequences(lines)
	return pad_sequences(tokenized, maxlen=length, padding='post')

def tokenize_it(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_length(lines):
	return max(len(line.split()) for line in lines)

def load_file(filename):
	return load(open(filename, 'rb'))

if __name__ == '__main__':
	model = load_model('model_128_0.h5')

	reader = easyocr.Reader(['en'])
	read_file = './example/Nobody likes war._43.jpg'

	dataset = load_file('eng_spa_all.pkl')
	train = load_file('eng_spa_train.pkl')
	test = load_file('eng_spa_test.pkl')

	eng_tokenizer = tokenize_it(dataset[:, 0])
	eng_vocab_size = len(eng_tokenizer.word_index) + 1
	eng_length = max_length(dataset[:, 0])

	es_tokenizer = tokenize_it(dataset[:, 1])
	es_vocab_size = len(es_tokenizer.word_index) + 1
	es_length = max_length(dataset[:, 1])

	test_string = reader.readtext(read_file, detail = 0, paragraph=True)

	if len(test_string) > 0:
		test_string = test_string[0]
		cont_test = np.asarray(test_string,dtype='<U275').reshape((1,))
		encoded = encode_sequences(eng_tokenizer, eng_length, cont_test)

		input_one = encoded.reshape(1,eng_length)
		input_two = np.zeros((1,es_length))

		predicted = []

		for i in range(0,es_length):

		    output = model.predict([input_one,input_two])
		    idx = np.argmax(output,axis=2)
		    idx_i = idx[0,i]

		    if i != es_length-1:
		        input_two[0,i+1] = idx_i

			if idx_i == 0:
		        break

		prediction = input_two[0].tolist()
		prediction.append(a)

		for el in prediction[1:]:
		    try:
		        predicted.append(es_tokenizer.index_word[el])
		    except:
		        predicted.append('')

		print('='*25)
		print('prediction:', " ".join(predicted))
		print('='*25)

	else:
		print('Couldn\'t read anything from the image. Please check if the input is in English.')
