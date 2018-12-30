import math
import winsound
import time

labels = ['a','a#','b','c','c#','d','d#','e','f','f#','g','g#']
# name is the complete name of a note (label + octave). the parameter
# n is the number of half-tone from A4 (e.g. D#1 is -42, A3 is -12, A5 is 12)
name   = lambda n: labels[n%len(labels)] + str(int((n+(9+4*12))/12))
# the frequency of a note. the parameter n is the number of half-tones
# from a4, which has a frequency of 440Hz, and is our reference note.
freq   = lambda n: int(440*(math.pow(2,1/12)**n))

# a dictionnary associating note frequencies to note names
notes  = {name(n): freq(n) for n in range(-42,60)}

# the period expressed in second, computed from a tempo in bpm
period = lambda tempo: 1/(tempo/60)

# play each note in sequence through the PC speaker at the given tempo
def play(song, tempo):
    for note in song.lower().split():
        if note in notes.keys():
            winsound.Beep(notes[note], int(period(tempo)*1000))
        else:
            time.sleep(period(tempo))

# "au clair de la lune"!! 'r' is a rest
play( 'c4 c4 C4 d4 e4 r d4 r c4 e4 d4 d4 c4 r r r '
      'c4 C4 c4 d4 e4 r d4 r c4 e4 d4 d4 c4 r r r '
      'd4 d4 d4 d4 A3 r a3 r d4 c4 B3 a3 g3 r r r '
      'c4 c4 c4 d4 e4 r d4 r c4 e4 d4 d4 c4 r r r ', 180 )