import numpy as np
import librosa
import librosa.display

# y : nd.ndarray
# sr : sampling rate
y, sr = librosa.load("트와이스.mp3")
librosa.feature.chroma_stft(y=y, sr=sr)             # Load an audio file as a floating point time series.
# array([[ 0.974,  0.881, ...,  0.925,  1.   ],
# [ 1.   ,  0.841, ...,  0.882,  0.878],
# ...,
# [ 0.658,  0.985, ...,  0.878,  0.764],
# [ 0.969,  0.92 , ...,  0.974,  0.915]])

# Use an energy (magnitude) spectrum instead of power spectrogram
# Short-time Fourier transform
# Returns a complex-valued matrix D such that
# np.abs(D[f, t]) is the maganitude of frequency bin f at frame t
# np.angle(D[f, t]) is the phase of frequency bin f at frame t

S = np.abs(librosa.stft(y))
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# array([[ 0.884,  0.91 , ...,  0.861,  0.858],
# [ 0.963,  0.785, ...,  0.968,  0.896],
# ...,
# [ 0.871,  1.   , ...,  0.928,  0.829],
# [ 1.   ,  0.982, ...,  0.93 ,  0.878]])

# Use a pre-computed power spectrogram with a larger frame

S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)
# array([[ 0.685,  0.477, ...,  0.961,  0.986],
# [ 0.674,  0.452, ...,  0.952,  0.926],
# ...,
# [ 0.844,  0.575, ...,  0.934,  0.869],
# [ 0.793,  0.663, ...,  0.964,  0.972]])

"""
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
#plt.axis('off')
#plt.axis('tight')
plt.tight_layout()
plt.show()
#plt.subplots_adjust(0,0,1,1,0,0)
#plt.savefig("happy_song.png",bbox_inches='tight', pad_inches=0)
"""
import matplotlib.pyplot as plt
plt.subplot(1, 1, 1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma)
plt.tight_layout()
plt.savefig("happy_song.png",bbox_inches='tight', pad_inches=0)