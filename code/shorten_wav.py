import scipy

address = r'C:\Users\tchase56\Documents\Startup\voice_data\wav\\'

[Fs, mixed] = scipy.io.wavfile.read(address + 'Yael_02.wav')
mixed = mixed[:Fs*60,0]

scipy.io.wavfile.write(address + 'Yael_02_shortened.wav', Fs, mixed)