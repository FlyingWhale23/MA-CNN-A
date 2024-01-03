import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 这会禁用所有可见的GPU
import matplotlib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
import numpy as np
from net import CNN

meta_file_pre = os.path.join("E:/Deepship/predict.csv")
meta_data_pre = pd.read_csv(meta_file_pre, header=0, encoding="gbk", names=["A","B","C","D","E","F","G"], usecols=["A","C"])
print(meta_data_pre)


def plot_confusion_matrix(cm, labels_name, title, colorbar=True, cmap=plt.cm.jet):
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)    # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.title(title)    # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

freq = 64
time = 94

predict_file = "./melsp_pre.npz"
# load test dataset
predict = np.load(predict_file)
x_predict = predict["x"]
# print(x_predict.shape)
y_predict = predict["y"]
x_predict = x_predict.reshape(2000,freq,time,1)
print('********************************')
# print(x_predict)
# print(x_predict[0].shape)
y_predict = to_categorical(y_predict, 4)
y_predict = np.argmax(y_predict,axis=1)

inputs = Input((64, 94, 1))
output = CNN(inputs)
model = Model(inputs, output)
# model.summary()
model.load_weights('./models_98.2/melsp+mcnna.hdf5')
pred = model.predict(x_predict)


result = np.argmax(pred, axis=1)
print(result)

df = pd.DataFrame({"img_path":meta_data_pre["A"], "tags":result})
df.to_csv("submit.csv",index=None,header=None)


target_names = ['0cargo', '1passenger', '2tanker', '3tug']
print(classification_report(y_predict, result, target_names=target_names, digits=5))
cm=confusion_matrix(y_predict, result)
print(cm)

plot_confusion_matrix(cm, ["cargo", "passenger ship", "oil tanker", "tug"], "Confusion Matrix")
plt.show()
