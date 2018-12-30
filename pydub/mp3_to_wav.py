
import pydub
from pydub import AudioSegment

duration = 60 * 3 * 1000

pydub.AudioSegment.converter = r"D:\ffmpeg\ffmpeg\bin\ffmpeg.exe"

sound = AudioSegment.from_mp3("D:\\MyPythonProject\\pydub\\happy_song.mp3")     # for happy_song.mp3
sound.export(r"D:\MyPythonProject\pydub\test_mood_wav.wav", format="wav")

sound = sound[:duration]
sound.export(r"D:\MyPythonProject\pydub\test_mood_wav_dutation.wav", format="wav")


sound = AudioSegment.from_mp3("D:\\MyPythonProject\\pydub\\test2_mood.mp3")  # for test2.mp3
sound.export(r"D:\MyPythonProject\pydub\test2_mood_wav.wav", format="wav")

sound = sound[:duration]
sound.export(r"D:\MyPythonProject\pydub\test2_mood_wav_duration.wav", format="wav")

sound = AudioSegment.from_mp3("D:\\MyPythonProject\\pydub\\test3.mp3")  # for test3.mp3
sound.export(r"D:\MyPythonProject\pydub\test3_mood_wav.wav", format="wav")

sound = sound[:duration]
sound.export(r"D:\MyPythonProject\pydub\test3_mood_wav_duration.wav", format="wav")