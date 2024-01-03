import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用所有可见的GPU
import matplotlib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from net import CNN

meta_file_pre = os.path.join("./predict.csv")
meta_data_pre = pd.read_csv(meta_file_pre, header=0, encoding="gbk", names=["A","B","C","D","E","F","G"], usecols=["A","C"])
print(meta_data_pre)

def plot_confusion_matrix(cm, labels_name, title, colorbar=True, cmap=plt.cm.jet):
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)
    plt.yticks(num_local, labels_name)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#
freq = 64
time = 94

predict_file = "melsp_pre.npz"
# load test dataset
predict = np.load(predict_file)
x_predict = predict["x"]
# print(x_predict.shape)
y_predict = predict["y"]
x_predict = x_predict.reshape(2000, freq, time, 1)
print('********************************')
y_predict = to_categorical(y_predict, 4)
y_predict = np.argmax(y_predict, axis=1)

inputs = Input((64, 94, 1))
output = CNN(inputs)
model = Model(inputs, output)
# model.summary()
model.load_weights('./models_98.2/MELSP+mcnna.hdf5')
pred = model.predict(x_predict)

# t = 0
# for i in range(100):
#    t0 = time.time()
#    out = model.predict(pre)
#    t += (time.time()-t0)
# print(t/100)

def h5_to_pb(model,save_path):

    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
#     for layer in layers:
#         print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=save_path,
                      name="model.pb",
                      as_text=False)

save_path = './PB/melsp+macnna'
if not os.path.exists(os.path.join(save_path,"model.pb")):
    h5_to_pb(model,save_path)


def init(pb_path):
    tf.Graph().as_default()
    config = tf.compat.v1.ConfigProto()
    output_graph_def = tf.compat.v1.GraphDef()
    with open(pb_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def,input_map=None,return_elements=None,name=None,op_dict=None,producer_op_list=None)

        sess = tf.compat.v1.Session(config=config)
    return sess

pre = x_predict[0:1, 0:64, 0:94, 0:1]

def process_image(handle=None,inputs=pre):
    model = handle
    inputss = model.graph.get_tensor_by_name("Input:0")
    output = model.graph.get_tensor_by_name("Identity:0")
    out = model.run(output, feed_dict={inputss: inputs})
    return out

model = init(os.path.join(save_path,"model.pb"))


import time

for i in range(10):
    out = process_image(model, inputs=pre)
t = 0
for i in range(100):
    t0 = time.time()
    out = process_image(model, inputs=pre)
    t += (time.time()-t0)

print("inference time：" + str((t / 100)))
print("inference speed/FPS："+str(int(1/(t/100))))
x_predict1 = x_predict[:1000, :, :, :]
x_predict2 = x_predict[1000:2000, :, :, :]
pred1 = process_image(model, inputs=x_predict1)
pred2 = process_image(model, inputs=x_predict2)
pred = np.concatenate([pred1, pred2], 0)
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
