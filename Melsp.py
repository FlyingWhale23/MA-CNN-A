import os
import librosa.display
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")


def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=16000)
    return x,fs

# # change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=512):
     stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
     log_stft = librosa.power_to_db(stft)
     melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
     return melsp

def show_wave(x):
    plt.plot(x)
    plt.show()

def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    # plt.colorbar()
    plt.show()


meta_file_tra = os.path.join("E:/Deepship/train.csv")
audio_dir_tra = os.path.join("E:/tongyuan/train/")

meta_file_val = os.path.join("E:/Deepship/validation.csv")
audio_dir_val = os.path.join("E:/tongyuan/validation/")

meta_file_pre = os.path.join("E:/Deepship/predict.csv")
audio_dir_pre = os.path.join("E:/tongyuan/eval/")

# load metadata
meta_data_tra = pd.read_csv(meta_file_tra, header=0, encoding="gbk", names=["A","B","C","D","E","F","G"], usecols=["A","C"])
print(meta_data_tra)

meta_data_val = pd.read_csv(meta_file_val, header=0, encoding="gbk", names=["A","B","C","D","E","F","G"], usecols=["A","C"])
print(meta_data_val)

meta_data_pre = pd.read_csv(meta_file_pre, header=0, encoding="gbk", names=["A","B","C","D","E","F","G"], usecols=["A","C"])
print(meta_data_pre)

# get data size
data_size_tra = meta_data_tra.shape
print(data_size_tra)
data_size_val = meta_data_val.shape
print(data_size_val)
data_size_pre = meta_data_pre.shape
print(data_size_pre)

x1 = list(meta_data_tra.loc[:,"A"])
# print(len(x1))
y1 = list(meta_data_tra.loc[:,"C"])

x2 = list(meta_data_val.loc[:,"A"])
y2 = list(meta_data_val.loc[:,"C"])

x3 = list(meta_data_pre.loc[:,"A"])
y3 = list(meta_data_pre.loc[:,"C"])



print("==========================melsp===========================")
x, fs = load_wave_data(audio_dir_tra, meta_data_tra.loc[0,"A"])
melsp = calculate_melsp(x)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x.shape, melsp.shape, fs))
show_wave(x)
show_melsp(melsp, fs)

print("==========================get training dataset ======================")
freq = 128
time = 94

# freq = 128
# time = 94


# freq = 128
# time = 47
# freq = 48
# time = 192



def save_np_data(filename, x, y, audio_dir, aug=None, rates=None):
    np_data = np.zeros(freq*time*len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data(audio_dir, x[i])
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
        np_targets[i] = y[i]
    np.savez(filename, x=np_data, y=np_targets)


if not os.path.exists("melsp_tra12894.npz"):
    save_np_data("melsp_tra12894.npz", x1,  y1, audio_dir=audio_dir_tra)

if not os.path.exists("melsp_val12894.npz"):
    save_np_data("melsp_val12894.npz", x2,  y2, audio_dir=audio_dir_val)

if not os.path.exists("melsp_pre12894.npz"):
    save_np_data("melsp_pre12894.npz", x3,  y3, audio_dir=audio_dir_pre)

