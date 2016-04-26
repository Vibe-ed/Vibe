#------------------------------------------------------------
# FastICA

# Notes about ICA
# 1. Ambiguous to permutations among microphones
# 2. Ambiguous to scaling of the signals (must normalize after
#    deconvolution
# 3. These are the only sources of ambiguity as long as the sources S_i are
#    not gaussian

import scipy.io.wavfile
import numpy as np
from sklearn.decomposition import FastICA

address = r'/Users/tylerchase/Documents/Startup/unmixing_voices/ICA/FastICA///'

mix = np.loadtxt(address + 'mix.dat') # Load mixed sources (source by time)
Fs = 11025 # Sampling frequency being used

# Normalize sources to have a maximum value of 1
normalizedMix = 0.99 * mix / (np.ones((np.shape(mix)[0],1)) * np.amax(np.absolute(mix),0))

# write the mixed source from each of the 5 microphones to a wav file
scipy.io.wavfile.write(address + 'mix1.wav', Fs, normalizedMix[:,0])
scipy.io.wavfile.write(address + 'mix2.wav', Fs, normalizedMix[:,1])
scipy.io.wavfile.write(address + 'mix3.wav', Fs, normalizedMix[:,2])
scipy.io.wavfile.write(address + 'mix4.wav', Fs, normalizedMix[:,3])
scipy.io.wavfile.write(address + 'mix5.wav', Fs, normalizedMix[:,4])

# Use FastICA algorith
ica = FastICA(n_components=5)
S = ica.fit_transform(normalizedMix)

# rescale each column to have maximum absolute value 1 
S = 0.99 * S / (np.ones((np.shape(mix)[0],1)) * np.amax(np.absolute(S),0))


# now have a listen --- You should have the following five samples:
# * Godfather
# * Southpark
# * Beethoven 5th
# * Austin Powers
# * Matrix (the movie, not the linear algebra construct :-) 

scipy.io.wavfile.write(address + 'unmix1.wav', Fs, S[:,0])
scipy.io.wavfile.write(address + 'unmix2.wav', Fs, S[:,1])
scipy.io.wavfile.write(address + 'unmix3.wav', Fs, S[:,2])
scipy.io.wavfile.write(address + 'unmix4.wav', Fs, S[:,3])
scipy.io.wavfile.write(address + 'unmix5.wav', Fs, S[:,4])
