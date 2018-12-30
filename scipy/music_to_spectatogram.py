import stft
import os
import scipy
import scipy.io.wavfile as wav
import matplotlib.pylab as pylab

def save_stft_image(source_filename, destination_filename):
    fs, audio = wav.read(source_filename)
    audio = scipy.mean(audio, axis=1)

    X = stft.spectrogram(audio)

    fig = pylab.figure()
    ax = pylab.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto', interpolation='nearest')
    pylab.savefig(destination_filename)

save_stft_image("test.wav","test.png")