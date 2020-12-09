import numpy as np

data = np.load('ocr_output.npz', allow_pickle=True)
predict = data['predict_dict']
print(predict)
