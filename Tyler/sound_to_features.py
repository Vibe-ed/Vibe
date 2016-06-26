import numpy as np
import scipy.io.wavfile
import scipy
from scipy.signal import spectrogram
from scipy.signal import get_window

def wav_to_frequencies(wav_path, nfft=None,  window_type='hann', sample_time=1, overlap=2):
    """
    converts wav to frequencies features.
    uses a spectogram with <sample_time> steps, and a <window_type> window.
    returns feature matrix X.
    """

    #calc overlap between samples.
    overlapFactor = float(sample_time)/overlap
    
    # Read in the 30 second samplings
    [Fs, wav] = scipy.io.wavfile.read(path)
    # Use built in spectrogram to compare to FFT
    [wav_f, wav_t, X] = spectrogram(wav,fs=Fs, window = get_window(window_type, Fs*sample_time), nperseg = Fs*sample_time, noverlap = Fs*overlapFactor) 

    return [wav_f, wav_t, X.T]





def main():
    path = '/Users/tomharamaty/Downloads/Yael_ny.wav'
    pass

if __name__ == '__main__':
    main()