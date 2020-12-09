import numpy as np

eng_words = []
hin_words = []

with  open('spa.txt') as fp:
    contents = fp.read()
    for line in contents.split('\n'):
        entry = line.split('\t')
        if len(entry) > 1:
            eng_words.append(entry[0][:-1])
            hin_words.append(entry[1][:-1])

print(len(eng_words))
np.savez('spa_eng_words.npz', swe_words=hin_words, eng_words=eng_words)
