__author__ = 'yaelcohen'

import pyaudio
import wave
import requests
import haven_speech

API_KEY_HAVEN = ""

FILE_NUM = 2
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "file"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print "recording..."
for i in range(FILE_NUM):
    frames = []
    for j in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    file_name = WAVE_OUTPUT_FILENAME+"_"+str(i)+".wav"
    waveFile = wave.open(file_name, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    # haven_speech.get_text(file_name)
print "finished recording"

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()