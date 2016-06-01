import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
from scipy.signal import get_window
from sklearn.linear_model import LogisticRegression

# Filter out any values below a certain sound threshold for training sets
def filter(matrix, times):
    filtered = []
    times_2 = []
    for i in range(np.shape(matrix)[1]):
        if np.sum(matrix[:,i])>30:
            filtered.append(matrix[:,i])
            times_2.append(times[i])
    filtered = np.array(filtered).T
    return(filtered, times_2)
    
def weighted_frequency(matrix, frequencies):
    volume = np.sum(matrix, axis = 0)
    temp = np.sum(matrix.T * frequencies, axis = 1)
    temp_2 = temp/volume
    print(volume[:3])
    print(temp[:3])
    print(temp_2[:3])
    return(temp_2)
    
#def weighted_frequency(matrix, frequencies):

# Address of the data files        
address = r'/Users/tylerchase/Documents/Startup/unmixing_voices/Voice_Samples/Tyler_Henry_Yael_IndividialVoices//'
address_2 = r'/Users/tylerchase/Documents/Startup/unmixing_voices/Voice_Samples/Tyler_Henry_Yael_IndividialVoices//'

# Read in the 10 second samplings
[Fs, Tyler] = scipy.io.wavfile.read(address + 'Tyler_in.wav')
[Fs, Henry] = scipy.io.wavfile.read(address + 'Henry_in.wav')
[Fs, Yael] = scipy.io.wavfile.read(address + 'Yael_in.wav')
    
Fs = Fs        

# Use built in spectrogram to compare to FFT
pointsFactor = 1.0
overlapFactor = 1.0/2
[Tyler_F, Tyler_t, Tyler_f_2] = spectrogram(Tyler[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = overlapFactor*Fs) 
[Henry_F, Henry_t, Henry_f_2] = spectrogram(Henry[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = overlapFactor*Fs) 
[Yael_F, Yael_t, Yael_f_2] = spectrogram(Yael[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = overlapFactor*Fs) 
None_f_2 = np.zeros((np.shape(Tyler_f_2)))

# Plot the magnitude spectrogram.
plt.figure()
plt.pcolormesh(Tyler_t, Tyler_F, Tyler_f_2).set_clim(0,0.010*10**6)
plt.xlim(Tyler_t[0],Tyler_t[-1])
plt.ylim(Tyler_F[0],Tyler_F[-1])
plt.title('Tyler')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.figure()
plt.pcolormesh(Henry_t, Henry_F, Henry_f_2).set_clim(0,0.010*10**6)
plt.xlim(Henry_t[0],Henry_t[-1])
plt.ylim(Henry_F[0],Henry_F[-1])
plt.title('Henry')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.figure()
plt.pcolormesh(Yael_t, Yael_F, Yael_f_2).set_clim(0,0.010*10**6)
plt.xlim(Yael_t[0],Yael_t[-1])
plt.ylim(Yael_F[0],Yael_F[-1])
plt.title('Yael')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.show()

# Plot average frequencies
Tyler_frequencies = weighted_frequency(Tyler_f_2, Tyler_F)
Henry_frequencies = weighted_frequency(Henry_f_2, Henry_F)
Yael_frequencies = weighted_frequency(Yael_f_2, Yael_F)


plt.figure()
plt.plot(Tyler_t, Tyler_frequencies, label = "Tyler")
plt.plot(Henry_t, Henry_frequencies, label = "Henry")
plt.plot(Yael_t, Yael_frequencies, label = "Yael")
plt.legend()
plt.show()
