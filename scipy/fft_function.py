# https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
fs, data = wavfile.read('happy_song.wav') # load the data
a = data.T[0] # this is a two channel soundtrack, I get the first track
plt.plot(a, 'r')
plt.show()

b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)
d = int(len(c)/2)  # you only need half of the fft list (real signal symmetry)

plt.plot(abs(c[:d-1]),'r')
plt.show()


fs, data = wavfile.read('트와이스.wav') # load the data
a = data.T[0] # this is a two channel soundtrack, I get the first track
plt.plot(a, 'r')
plt.show()

b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)
d = int(len(c)/2)  # you only need half of the fft list (real signal symmetry)
plt.plot(abs(c[:d-1]),'r')
plt.show()