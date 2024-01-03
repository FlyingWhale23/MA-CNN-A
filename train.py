import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from net import CNN
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from keras.models import Model
from keras_flops import get_flops #这里！
from tensorflow.keras import Input, models
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


freq = 64
time = 94

train_files = ["./melsp_tra.npz"]
test_file = "./melsp_val.npz"

train_num = 14000
test_num = 4000

# define dataset placeholders
x_train = np.zeros(freq*time*train_num*len(train_files)).reshape(train_num*len(train_files), freq, time)
y_train = np.zeros(train_num*len(train_files))

# load dataset
for i in range(len(train_files)):
    data = np.load(train_files[i])
    x_train[i*train_num:(i+1)*train_num] = data["x"]
    y_train[i*train_num:(i+1)*train_num] = data["y"]

# load test dataset
test_data = np.load(test_file)
x_test = test_data["x"]
y_test = test_data["y"]


print("=====================reshape training dataset====================")
# redefine target data into one hot vector
classes = 4
y_train = tf.keras.utils.to_categorical(y_train, classes)
y_test = tf.keras.utils.to_categorical(y_test, classes)
# print(y_test)
# reshape training dataset
x_train = x_train.reshape(train_num*1, freq, time, 1)
x_test = x_test.reshape(test_num, freq, time, 1)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(x_train.shape,
                                                                y_train.shape,
                                                                x_test.shape,
                                                                y_test.shape))


inputs = Input((64, 94, 1))
output = CNN(inputs)
model = Model(inputs, output)
# model.summary()
# plot_model(model, 'SCNN_model.png', show_shapes=True)
# print(get_flops(model))

print("=====================Optimization and callbacks======================")
# initiate Adam optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6, amsgrad=True)
# Let's train the model using Adam with amsgrad
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# directory for model checkpoints
model_dir = "./save_model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


# early stopping and model checkpoint# earl
es_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,verbose=1, mode='min')
chkpt = os.path.join(model_dir, 'macnna.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.6,patience=3,verbose=1,mode="min")

print("=====================Train CNN model with between class dataset======================")
# between class data generator
class MixupGenerator():
    def __init__(self, x_train, y_train, batch_size=16, alpha=0.2, shuffle=True):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(x_train)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                x, y = self.__data_generation(batch_ids)

                yield x, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.x_train.shape
        _, class_num = self.y_train.shape
        x1 = self.x_train[batch_ids[:self.batch_size]]
        x2 = self.x_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        x_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        x = x1 * x_l + x2 * (1 - x_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return x, y
# train model
batch_size = 16
epochs = 200

training_generator = MixupGenerator(x_train, y_train)()
history = model.fit_generator(generator=training_generator,
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    callbacks=[es_cb, cp_cb, reduce_lr])


evaluation = model.evaluate(x_test, y_test)
print(evaluation)

acc = history.history['accuracy']
# np.savez("./macnna/melsp_acc.npz",acc)
val_acc = history.history['val_accuracy']
# np.savez("./macnna/CRNN/melsp_val_acc.npz",val_acc)
loss = history.history['loss']
# np.savez("./macnna/CRNN/melsp_loss.npz",loss)
val_loss = history.history['val_loss']
# np.savez("./macnna/CRNN/melsp_val_loss.npz",val_loss)
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.figure()



plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
