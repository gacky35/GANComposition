# %%
import musdb
from torch import sign
mus = musdb.DB(download=True)
display(mus[0].stems)
# %%
sample = mus[0].stems[1:4]
# %%
from scipy.io.wavfile import write
samplerate = 44100
write("./sample_recover.wav", samplerate, y_inv)
write("./sample.wav", samplerate, mus[0].stems[1][:,0])
# %%
import matplotlib.pyplot as plt
# %%
plt.plot(mus[0].stems[0])
# %%
from scipy import signal
fs = 10e3
f,t,Sxx=signal.spectrogram(sample[1][:,0], fs)
# %%
import numpy as np
plt.pcolormesh(t, f, 10*np.log(Sxx))
plt.show()
# %%
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise
x.shape
# %%
plt.plot(x)
# %%
mixture = mus[0].stems[0]
separate = mus[0].stems[1:]
# %%
print(mixture[0], separate[0][0]+separate[1][0]+separate[2][0]+separate[3][0])
# %%
base_sound = separate[0] + separate[1]
plt.plot(base_sound)
# %%
base_sound_monoral = (base_sound[:,0]+base_sound[:,1]) / 2
plt.plot(base_sound_monoral)
# %%
from tqdm import tqdm
for i in tqdm(range(len(mus))):
    base_sound_ndarray = mus[i].stems[1:3]
    base_sound = base_sound_ndarray[0] + base_sound_ndarray[1]
    base_sound_monoral = (base_sound[:,0] + base_sound[:,1]) / 2
    fs = 10e3
    f,t,Sxx=signal.spectrogram(base_sound_monoral, fs)
    plt.pcolormesh(t, f, 10*np.log(Sxx))
    plt.savefig(f"./spectrograms/spectrogram{i:03d}.png")
# %%
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
from IPython.display import display
import pandas as pd
import librosa
import librosa.display
import musdb
from torch import sign
import matplotlib.pyplot as plt
# %%
mus = musdb.DB(download=True)
# %%
audio_path = librosa.util.example_audio_file(); audio_path
y, sr = librosa.load(audio_path)
display(mus[0].stems)
# %%
sr = 44100
librosa.display.waveplot(y, sr=sr)
# %%
(y.shape, mus[0].audio.shape)
# %%
def min_max(x, axis=None):
    min_num = x.min(axis=axis,keepdims=True)
    max_num = x.max(axis=axis,keepdims=True)
    result = (x-min_num) / (max_num-min_num)
    return result
# %%
def decord_min_max(x, min_num, max_num, axis=None):
    result = x*max_num-x*min_num+min_num
    return result
# %%
D = librosa.stft(mus[0].stems[1][:,0])  # STFT
S, phase = librosa.magphase(D)  # 複素数を強度と位相へ変換
Sdb = librosa.amplitude_to_db(S)  # 強度をdb単位へ変換
# print(Sdb)
# print(Sdb.shape)
min_num = Sdb.min(axis=None,keepdims=True)
max_num = Sdb.max(axis=None,keepdims=True)
# print(min_num, max_num)
fig_size = np.array(Sdb.shape)
dpi = 100
plt.subplots(figsize=fig_size/dpi, dpi=dpi)
Sdb = min_max(Sdb) * 255 # db単位を画像で扱える範囲に正規化
# print(Sdb)
# print(Sdb.shape)
# print(max(list(map(lambda x: max(x), Sdb))), min(list(map(lambda x: min(x), Sdb))))
librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='log')  # スペクトログラムを表示
# im = librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='log')  # スペクトログラムを表示
# D2 = im.get_array().reshape(im._meshHeight, im._meshWidth)
plt.axis('off')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
# plt.show()
# import cv2
# cv2.imwrite('./cv2_sample.png', D2)
plt.savefig("./sample_normal.png")
from PIL import Image
img = Image.open("./sample_normal.png")
gray = img.convert('L')
gray.save('./sample_normal_gray.png')
#%%
Sdb = librosa.amplitude_to_db(S)
display(Sdb)
min_num = Sdb.min(axis=None,keepdims=True)
max_num = Sdb.max(axis=None,keepdims=True)
print(min_num, max_num)
display(min_max(Sdb))
display(decord_min_max(min_max(Sdb), min_num, max_num))
# %%
mus[0].stems[1][:,0].shape
# %%
p = pd.DataFrame(phase.flatten()).sample(n=1000)  # 散布図にするにはデータ数多すぎなので、1000データをランダムサンプリング
mpl_collection = plt.scatter(np.real(p), np.imag(p))
mpl_collection.axes.set(title="複素平面上の位相の散布図", xlabel="実部", ylabel="虚部", aspect='equal')
# %%
D = S * np.exp(1j*phase)  # 直交形式への変換はlibrosaの関数ないみたいなので、自分で計算する。
y_inv = librosa.istft(D)
# display(IPython.display.Audio(y_inv, rate=sr))
mpl_collection = librosa.display.waveplot(y_inv, sr=sr)
mpl_collection.axes.set(title="位相情報を使って復元した音声波形", ylabel="波形の振幅")
# %%
librosa.display.waveplot(mus[0].stems[1][:,0])
# %%
y_inv
# %%
import soundfile as sf
sf.write("./sample_recover.wav", y_inv, sr)
sf.write("./sample.wav", mus[0].stems[1][:,0], sr)
# %%
def griffinlim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window)

    return inverse
# %%
import cv2
# %%
img = cv2.imread('./sample_normal_gray.png', cv2.IMREAD_GRAYSCALE)
# img = img.reshape(1025, 587)
print(img.shape)
plt.gray()
plt.imshow(img)
print(img)
# %%
sdb = pd.DataFrame(img.flatten())
sdb.hist(bins=20, range=(sdb.min().values[0], sdb.quantile(0.9).values[0]) ) # Sの最小値～90%点までの分布を表示
# %%
img[::-1]
# %%
img_decord = decord_min_max(img[::-1] / 255, -48.34430262, 31.65569738)
# %%
img_decord = img_decord.reshape(1025, 587)
# %%
min_num = (img_decord).min(axis=None,keepdims=True)
max_num = (img_decord).max(axis=None,keepdims=True)
# %%
print(min_num, max_num)
# %%
reverse = librosa.griffinlim(img_decord)
display(reverse)
# %%
D = img_decord * np.exp(np.zeros(img_decord.shape))
y_inv = librosa.istft(D)
# display(IPython.display.Audio(y_inv, rate=sr))
mpl_collection = librosa.display.waveplot(y_inv, sr=sr)
# mpl_collection.axes.set(title="位相０で復元した音声波形", ylabel="波形の振幅")
# mpl_collection.axes.set_ylim(ylim);
# %%
mpl_collection = librosa.display.waveplot(reverse, sr=sr)
# %%
y_inv = griffinlim(img_decord)
mpl_collection = librosa.display.waveplot(y_inv, sr=sr)
# %%
print(min(mus[0].stems[1][:,0]), max(mus[0].stems[1][:,0]))

# %%
print(min(y_inv), max(y_inv))
# %%
sdb = pd.DataFrame(mus[0].stems[1][:,0].flatten())
sdb.hist(bins=20, range=(sdb.min().values[0], sdb.quantile(0.9).values[0]) ) # Sの最小値～90%点までの分布を表示
plt.title("dB単位の強度の分布")
# %%
