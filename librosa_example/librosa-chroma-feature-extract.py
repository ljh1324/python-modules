import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment

def segmentation_mp3(load_file, save_file):
    sound = AudioSegment.from_mp3(load_file)  # for happy_song.mp3
    duration = 60 * 1000
    sound[:duration].export(save_file + '-begin.mp3', format="mp3")
    sound[duration:duration * 2].export(save_file + '-middle.mp3', format="mp3")
    sound[-duration:].export(save_file + '-end.mp3', format="mp3")

def extract_feature(file_name, save_file_name):
    # y : nd.ndarray
    # sr : sampling rate
    y, sr = librosa.load(file_name)
    #chroma = librosa.feature.chroma_stft(y=y, sr=sr)             # Load an audio file as a floating point time series.
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

    #s = np.abs(librosa.stft(y))
    #chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    # array([[ 0.884,  0.91 , ...,  0.861,  0.858],
    # [ 0.963,  0.785, ...,  0.968,  0.896],
    # ...,
    # [ 0.871,  1.   , ...,  0.928,  0.829],
    # [ 1.   ,  0.982, ...,  0.93 ,  0.878]])

    # Use a pre-computed power spectrogram with a larger frame
    s = np.abs(librosa.stft(y, n_fft=4096))**2
    chroma = librosa.feature.chroma_stft(S=s, sr=sr)    # S: power spectrogram

    # array([[ 0.685,  0.477, ...,  0.961,  0.986],
    # [ 0.674,  0.452, ...,  0.952,  0.926],
    # ...,
    # [ 0.844,  0.575, ...,  0.934,  0.869],
    # [ 0.793,  0.663, ...,  0.964,  0.972]])

    plt.subplot(1, 1, 1)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma)
    plt.tight_layout()
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)

segmentation_mp3('Breath.mp3', 'Breath')
extract_feature('Breath-begin.mp3', 'Breath0.png')
extract_feature('Breath-middle.mp3', 'Breath1.png')
extract_feature('Breath-end.mp3', 'Breath2.png')

segmentation_mp3('Hush.mp3', 'Hush')
extract_feature('Hush-begin.mp3', 'Hush0.png')
extract_feature('Hush-middle.mp3', 'Hush1.png')
extract_feature('Hush-end.mp3', 'Hush2.png')

segmentation_mp3('LastDance.mp3', 'LastDance')
extract_feature('LastDance-begin.mp3', 'LastDance0.png')
extract_feature('LastDance-middle.mp3', 'LastDance1.png')
extract_feature('LastDance-end.mp3', 'LastDance2.png')

segmentation_mp3('HelloBitch.mp3', 'HelloBitch')
extract_feature('HelloBitch-begin.mp3', 'HelloBitch0.png')
extract_feature('HelloBitch-middle.mp3', 'HelloBitch1.png')
extract_feature('HelloBitch-end.mp3', 'HelloBitch2.png')