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
import soundfile as sf
# %%
mus = musdb.DB(root='E:\hackathon\musdb18')
# %%
musics = mus.load_mus_tracks(subsets=None, split=None)
# %%
sr = 44100
for i in tqdm(range(len(musics))):
    base_sounds_stereo = musics[i].stems[1:3]
    base_sound_stereo = base_sounds_stereo[0] + base_sounds_stereo[1]
    base_sound = (base_sound_stereo[:,0] + base_sound_stereo[:,1]) / 2
    j = 0
    with tqdm() as pbar:
        while j <= len(base_sound):
            D = librosa.stft(base_sound[j:j+sr*10])  # STFT
            S, phase = librosa.magphase(D)  # 複素数を強度と位相へ変換
            Sdb = librosa.amplitude_to_db(S)  # 強度をdb単位へ変換
            plt.rcParams['image.cmap'] = 'inferno'
            librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='log')  # スペクトログラムを表示
            plt.savefig(f"./spectrograms/spectrogram{i:03d}_{j:03d}.png")
            pbar.update(1)
            j += sr * 10
# %%
