import easyocr
import os
import numpy as np

path = './trdg/out/'
files = os.listdir(path)

data = np.load('./trdg/spa_eng_words.npz')
eng_words = data['eng_words']
#hin_words = data['spa_words']

predict = dict()

reader = easyocr.Reader(['en'])

for i, el in enumerate(files):
    print(i+1)
    result = reader.readtext(path+el, detail = 0, paragraph=True)

    if len(result) > 0:
        if result[0] not in predict:
            predict[result[0]] = el.split('_')[0]

np.savez('ocr_output_spa.npz', predict_dict=predict)
print(len(predict))

data = np.load('ocr_output_spa.npz', allow_pickle=True)
predict = data['predict_dict'][()]

import pandas as pd

df = pd.DataFrame(predict.items())
df['extra'] = 'lol'

df.to_csv('./generated.txt', sep='\t', header=False, index=False)

# print(df)
