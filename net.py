from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D,Dropout
from tensorflow.keras.layers import BatchNormalization, Add
from keras.utils.vis_utils import plot_model
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Model, layers
import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda
from keras_flops import get_flops
# se注意力机制
def se_block(inputs, ratio=4):  # ratio代表第一个全连接层下降通道数的系数
    # 获取输入特征图的通道数
    in_channel = inputs.shape[-1]
    # 全局平均池化[h,w,c]==>[None,c]
    x = layers.GlobalAveragePooling2D()(inputs)
    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=(1, 1, in_channel))(x)
    # [1,1,c]==>[1,1,c/4]
    x = layers.Dense(in_channel // ratio)(x)  # 全连接下降通道数
    # relu激活
    x = tf.nn.relu(x)
    # [1,1,c/4]==>[1,1,c]
    x = layers.Dense(in_channel)(x)  # 全连接上升通道数
    # sigmoid激活，权重归一化
    x = tf.nn.sigmoid(x)
    # [h,w,c]*[1,1,c]==>[h,w,c]
    outputs = layers.multiply([inputs, x])  # 归一化权重和原输入特征图逐通道相乘
    return outputs

classes = 4

def eca_block(inputs, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = inputs.shape[-1]
    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    # print(kernel_size)
    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    # [h,w,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling2D()(inputs)
    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)
    # [c,1]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    # sigmoid激活
    x = tf.nn.sigmoid(x)

    # [c,1]==>[1,1,c]
    x = layers.Reshape((1, 1, in_channel))(x)
    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])
    return outputs



def cba(inputs, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def CNN(inputs):
    x_1 = cba(inputs, filters=32, kernel_size=(1,8), strides=(1,2))
    x_1 = cba(x_1, filters=32, kernel_size=(8,1), strides=(2,1))
    x_1 = cba(x_1, filters=64, kernel_size=(1,8), strides=(1,2))
    in_channel11 = x_1.shape[-1]
    kernel_size = int(abs((math.log(in_channel11, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w11 = layers.GlobalAveragePooling2D()(x_1)
    x_w11 = layers.Reshape(target_shape=(in_channel11, 1))(x_w11)
    x_w11 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w11)
    x_w11 = tf.nn.sigmoid(x_w11)
    x_w11 = layers.Reshape((in_channel11,))(x_w11)

    x_1 = cba(x_1, filters=64, kernel_size=(8,1), strides=(2,1))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w12 = layers.GlobalAveragePooling2D()(x_1)
    x_w12 = layers.Reshape(target_shape=(in_channel11, 1))(x_w12)
    x_w12 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w12)
    x_w12 = tf.nn.sigmoid(x_w12)
    x_w12 = layers.Reshape((64,))(x_w12)


    x_2 = cba(inputs, filters=32, kernel_size=(1,16), strides=(1,2))
    x_2 = cba(x_2, filters=32, kernel_size=(16,1), strides=(2,1))
    x_2 = cba(x_2, filters=64, kernel_size=(1,16), strides=(1,2))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w21 = layers.GlobalAveragePooling2D()(x_2)
    x_w21 = layers.Reshape(target_shape=(in_channel11, 1))(x_w21)
    x_w21 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w21)
    x_w21 = tf.nn.sigmoid(x_w21)
    x_w21 = layers.Reshape((64,))(x_w21)

    x_2 = cba(x_2, filters=64, kernel_size=(16, 1), strides=(2, 1))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w22 = layers.GlobalAveragePooling2D()(x_2)
    x_w22 = layers.Reshape(target_shape=(in_channel11, 1))(x_w22)
    x_w22 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w22)
    x_w22 = tf.nn.sigmoid(x_w22)
    x_w22 = layers.Reshape((64,))(x_w22)



    x_3 = cba(inputs, filters=32, kernel_size=(1,32), strides=(1,2))
    x_3 = cba(x_3, filters=32, kernel_size=(32,1), strides=(2,1))
    x_3 = cba(x_3, filters=64, kernel_size=(1,32), strides=(1,2))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w31 = layers.GlobalAveragePooling2D()(x_3)
    x_w31 = layers.Reshape(target_shape=(in_channel11, 1))(x_w31)
    x_w31 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w31)
    x_w31 = tf.nn.sigmoid(x_w31)
    x_w31 = layers.Reshape((64,))(x_w31)

    x_3 = cba(x_3, filters=64, kernel_size=(32, 1), strides=(2, 1))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w32 = layers.GlobalAveragePooling2D()(x_3)
    x_w32 = layers.Reshape(target_shape=(in_channel11, 1))(x_w32)
    x_w32 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w32)
    x_w32 = tf.nn.sigmoid(x_w32)
    x_w32 = layers.Reshape((64,))(x_w32)


    x_4= cba(inputs, filters=32, kernel_size=(1,64), strides=(1,2))
    x_4 = cba(x_4, filters=32, kernel_size=(64,1), strides=(2,1))
    x_4 = cba(x_4, filters=64, kernel_size=(1,64), strides=(1,2))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w41 = layers.GlobalAveragePooling2D()(x_4)
    x_w41 = layers.Reshape(target_shape=(in_channel11, 1))(x_w41)
    x_w41 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w41)
    x_w41 = tf.nn.sigmoid(x_w41)
    x_w41 = layers.Reshape((64,))(x_w41)
    x_4 = cba(x_4, filters=64, kernel_size=(64, 1), strides=(2, 1))
    kernel_size = int(abs((math.log(64, 2) + 1) / 2))
    if kernel_size % 2:
        kernel_size = kernel_size
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
    x_w42 = layers.GlobalAveragePooling2D()(x_4)
    x_w42= layers.Reshape(target_shape=(in_channel11, 1))(x_w42)
    x_w42 = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x_w42)
    x_w42 = tf.nn.sigmoid(x_w42)
    x_w42 = layers.Reshape((64,))(x_w42)



    x = Add()([x_1, x_2, x_3, x_4])
    x_w = Add()([x_w11, x_w12, x_w21, x_w22, x_w31, x_w32, x_w41, x_w42])

    x = layers.multiply([x, x_w])

    x = cba(x, filters=98, kernel_size=(1,8), strides=(1,2))
    x = cba(x, filters=98, kernel_size=(8,1), strides=(2,1))

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    return x

# inputs = Input((64, 94, 1))
# output = CNN(inputs)
# model = Model(inputs, output)
# model.summary()
# plot_model(model, 'CNN_model.png', show_shapes=True)
# print(get_flops(model))



