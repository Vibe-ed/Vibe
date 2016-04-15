#------------------------------------------------------------
# ICA

# Notes about ICA
# 1. Ambiguous to permutations among microphones
# 2. Ambiguous to scaling of the signals (must normalize after
#    deconvolution
# 3. These are the only sources of ambiguity as long as the sources S_i are
#    not gaussian

import scipy.io.wavfile
import numpy as np

address = r'/Users/tylerchase/Documents/Startup/unmixing_voices/ICA///'

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

W = np.identity(5) # Initialize unmixing matrix

# this is the annealing schedule I used for the learning rate.
# (We used stochastic gradient descent, where each value in the 
# array was used as the learning rate for one pass through the data.)
# Note: If this doesn't work for you, feel free to fiddle with learning
# rates, etc. to make it work.
anneal = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001])

for i in range(len(anneal)):
    samples = np.shape(mix)[0] # number of time steps in sound signal
    permute = np.random.permutation(samples) # permute the time steps to get a more random distribution for the convergence of W
    for j in range(samples): # iterate over total time steps in the sound signal
        x = np.transpose(mix[permute[j],:]) # define five mixed samples at a given time step
        g = 1 / (1 + np.exp(np.dot(-W, x))) # define sigmoid function as function of W and x
        W = W + anneal[i] * (np.outer((1-2*g), np.transpose(x)) + np.transpose(np.linalg.inv(W))) # iterate over W to approach deconvolution matrix
        
# After finding W, use it to unmix the sources
S = np.transpose(np.dot(W, np.transpose(mix)))

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


