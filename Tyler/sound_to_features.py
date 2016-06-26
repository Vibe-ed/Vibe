import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy
from scipy.signal import spectrogram
from scipy.signal import get_window

def wav_to_frequencies(wav_path, nfft=None,  window_type='hamming', sample_time=0.020, overlap=0.015):
    """
    converts wav to frequencies features.
    uses a spectogram with <sample_time> steps, and a <window_type> window.
    returns arr:
    arr[0] - frequencies.
    arr[1] - seconds.
    arr[2] - feature matrix X.
    
    """
<<<<<<< HEAD
    
=======


    #calc overlap between samples.
    overlapFactor = float(sample_time)/overlap
    #overlapFactor = 0
>>>>>>> f68a3da5769c2317f7668a8c345780f1f31ab43e
    # Read in the 30 second samplings
    [Fs, wav] = scipy.io.wavfile.read(wav_path)
    # Use built in spectrogram to compare to FFT
    [wav_f, wav_t, X] = spectrogram(wav,
                                    fs=Fs,
                                    window = get_window(window_type,
                                    Fs*sample_time),
                                    nperseg = Fs*sample_time,
                                    noverlap = Fs*overlap,
                                    scaling = 'spectrum'
                                    ) 

    return [wav_f, wav_t, X.T]


def main():
    path = '/Users/tomharamaty/Downloads/Yael_ny.wav'
    [Fs, wav] = scipy.io.wavfile.read(path)
    plt.plot(wav)
    i=0
    print(int(2.0*len(wav)/Fs))
    for i in range(int(2.0*len(wav)/Fs)):
        plt.axvline(float(i)*Fs/2,color='r',linewidth=1)
    plt.show()
if __name__ == '__main__':
    main()