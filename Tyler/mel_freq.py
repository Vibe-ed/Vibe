import numpy as np
from scipy.fftpack import dct
import scipy
import math
from sound_to_features import wav_to_frequencies
import matplotlib.pyplot as plt




# Address of the data files        
address = r'/Users/tylerchase/repos/Vibe/voice_data/car_recordings//'

[Fs, Henry] = scipy.io.wavfile.read(address + 'Henry_abc.wav')


[Spectrogram_F, Spectrogram_t, Spectrogram] = wav_to_frequencies(address + 'Henry_abc.wav') 


numCoefficients = 10 # choose the sive of mfcc array
minHz = 300.0
maxHz = 8000.0  
nfft = np.shape(Spectrogram)[1]

def melFilterBank(blockSize):
    numBands = numCoefficients
    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    # Create a matrix for triangular filters, one row per filter
    filterMatrix = np.zeros((numBands, blockSize))
    melRange = np.array(range(numBands + 2))
    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel
    freqCenterFilters = melToFreq(melCenterFilters)
    binCenterFilters = np.floor((nfft*2 + 1) * freqCenterFilters / Fs).astype(int)

    #plt.figure()
    for i in range(numBands):
        start, center, end = binCenterFilters[i:i + 3]
        k1 = np.float32(center - start)
        k2 = np.float32(end - center)
        up = (np.array(range(start, center)) - start) / k1
        down = (end - np.array(range(center, end))) / k2
        filterMatrix[i][start:center] = up
        filterMatrix[i][center:end] = down       
        #plt.plot(Spectrogram_F, filterMatrix[i])
    return filterMatrix.transpose()
    #plt.show()
    

def freqToMel(freq):
    return 1125.0 * np.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (np.exp(mel / 1125.0)-1)
    
def spectrogram_to_mel(spectrogram):
    mel_matrix = []
    for i in range(1):    
        powerSpectrum = spectrogram[i,:]
        plt.figure()
        plt.plot(powerSpectrum)
        filteredSpectrum = np.dot(powerSpectrum, melFilterBank(len(powerSpectrum)))
        plt.figure()
        plt.plot(melFilterBank(len(powerSpectrum)))
        plt.figure()
        plt.plot(filteredSpectrum)
        logSpectrum = np.log(filteredSpectrum)
        dctSpectrum = dct(logSpectrum, type=2)  # MFCC :)
        mel_matrix.append(dctSpectrum)
    return(np.array(mel_matrix))
    
#mel = spectrogram_to_mel(Spectrogram)
    
mel_matrix = []
for i in range(1):    
    powerSpectrum = Spectrogram[1500,:]
    plt.figure()
    plt.plot(Spectrogram_F, powerSpectrum)
    filteredSpectrum = np.dot(powerSpectrum, melFilterBank(len(powerSpectrum)))
    plt.figure()
    plt.plot(Spectrogram_F, melFilterBank(len(powerSpectrum)))
    plt.figure()
    plt.plot(filteredSpectrum)
    logSpectrum = np.log(filteredSpectrum)
    dctSpectrum = dct(logSpectrum, type=2)  # MFCC :)
    plt.figure()
    plt.plot(dctSpectrum)
    mel_matrix.append(dctSpectrum[1:14])
    
plt.show()
