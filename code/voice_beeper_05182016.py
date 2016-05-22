import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
from scipy.signal import get_window
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

# Filter out any values below a certain sound threshold for training sets
def filter(matrix, times):
    filtered = []
    times_2 = []
    for i in range(np.shape(matrix)[1]):
        if np.sum(matrix[:,i])>150:
            filtered.append(matrix[:,i])
            times_2.append(times[i])
    filtered = np.array(filtered).T
    return(filtered, times_2)

# Address of the data files        
address = r'C:\Users\tchase56\Documents\Startup\voice_data\wav\\'
address_2 = r'C:\Users\tchase56\Documents\Startup\voice_data\wav\\'

# Read in the 30 second samplings
[Fs, Tyler] = scipy.io.wavfile.read(address + 'Tyler_ind_02.wav')
[Fs, Henry] = scipy.io.wavfile.read(address + 'Henry_ind_02.wav')
[Fs, Yael] = scipy.io.wavfile.read(address + 'Yael_ind_02.wav')
# Read in the 8 second samplings
[Fs, Tyler_test] = scipy.io.wavfile.read(address + 'Tyler_ind_01.wav')
[Fs, Henry_test] = scipy.io.wavfile.read(address + 'Henry_ind_01.wav')
[Fs, Yael_test] = scipy.io.wavfile.read(address + 'Yael_ind_01.wav')
# Read in the mixed sample
[Fs_2, Mixed] = scipy.io.wavfile.read(address + 'Yael_02_shortened.wav')
    
# Use built in spectrogram to compare to FFT
pointsFactor = 1.0
[Tyler_F, Tyler_t, Tyler_f_2] = spectrogram(Tyler[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Henry_F, Henry_t, Henry_f_2] = spectrogram(Henry[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Yael_F, Yael_t, Yael_f_2] = spectrogram(Yael[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Tyler_F_test, Tyler_t_test, Tyler_f_2_test] = spectrogram(Tyler_test[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Henry_F_test, Henry_t_test, Henry_f_2_test] = spectrogram(Henry_test[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Yael_F_test, Yael_t_test, Yael_f_2_test] = spectrogram(Yael_test[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Mixed_F, Mixed_t, Mixed_f_2] = spectrogram(Mixed,Fs_2, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
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


# Filter out time points with no sound
[Tyler_filtered, Tyler_t_filtered] = filter(Tyler_f_2, Tyler_t)
[Henry_filtered, Henry_t_filtered] = filter(Henry_f_2, Henry_t)
[Yael_filtered, Yael_t_filtered] = filter(Yael_f_2, Yael_t)
[Tyler_filtered_test, Tyler_t_filtered_test] = filter(Tyler_f_2_test, Tyler_t_test)
[Henry_filtered_test, Henry_t_filtered_test] = filter(Henry_f_2_test, Henry_t_test)
[Yael_filtered_test, Yael_t_filtered_test] = filter(Yael_f_2_test, Yael_t_test)
None_t = Tyler_t
Tyler_t = Tyler_t_filtered
Henry_t = Henry_t_filtered
Yael_t = Yael_t_filtered
Tyler_f_2 = Tyler_filtered
Henry_f_2 = Henry_filtered
Yael_f_2 = Yael_filtered



# Form y data
None_y = np.zeros(np.shape(None_f_2)[1])
Tyler_y = np.ones(np.shape(Tyler_f_2)[1])
Henry_y = np.ones(np.shape(Henry_f_2)[1])*2
Yael_y = np.ones(np.shape(Yael_f_2)[1])*3
y = np.matrix(np.concatenate((None_y,Tyler_y,Henry_y,Yael_y)))
#test
Tyler_y_test = np.ones(np.shape(Tyler_filtered_test)[1])
Henry_y_test = np.ones(np.shape(Henry_filtered_test)[1])*2
Yael_y_test = np.ones(np.shape(Yael_filtered_test)[1])*3

# Form x data
samples = np.shape(Tyler_f_2)[1] + np.shape(Henry_f_2)[1] + np.shape(Yael_f_2)[1] + np.shape(None_f_2)[1]
variables = np.shape(Tyler_f_2)[0]
x = np.concatenate((None_f_2,Tyler_f_2,Henry_f_2,Yael_f_2), axis = 1)

# randomize x and y
temp = np.concatenate((y,x), axis = 0).T
np.random.seed([131])
np.random.shuffle(temp)
temp = temp.T
y_shuffled = np.matrix((temp[0,:])).T
y_shuffled = np.array(y_shuffled)
y_shuffled = y_shuffled.reshape((samples,))
x_shuffled = np.matrix(temp[1:,:]).T

# Plot fit of Training
logistic_full = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 1)
logistic_full.fit(x_shuffled, y_shuffled)
# Training Error
Tyler_predict = logistic_full.predict(Tyler_f_2.T)
Henry_predict = logistic_full.predict(Henry_f_2.T)
Yael_predict = logistic_full.predict(Yael_f_2.T)
None_predict = logistic_full.predict(None_f_2.T)
Tyler_score = logistic_full.score(Tyler_f_2.T, Tyler_y)
Henry_score = logistic_full.score(Henry_f_2.T, Henry_y)
Yael_score = logistic_full.score(Yael_f_2.T, Yael_y)
None_score = logistic_full.score(None_f_2.T, None_y)
plt.figure()
plt.title('Yael Training Prediction' + '\n' + str(Yael_score))
plt.scatter(Yael_t, Yael_predict)
plt.ylim(-1,4)
plt.figure()
plt.title('Henry Training Prediction' + '\n' + str(Henry_score))
plt.scatter(Henry_t, Henry_predict)
plt.ylim(-1,4)
plt.figure()
plt.title('Tyler Training Prediction' + '\n' + str(Tyler_score))
plt.scatter(Tyler_t, Tyler_predict)
plt.ylim(-1,4)
plt.figure()
plt.title('Silence Training Prediction' + '\n' + str(None_score))
plt.scatter(None_t, None_predict)
plt.ylim(-1,4)
plt.show()

# Test Error
Tyler_predict_test = logistic_full.predict(Tyler_filtered_test.T)
Henry_predict_test = logistic_full.predict(Henry_filtered_test.T)
Yael_predict_test = logistic_full.predict(Yael_filtered_test.T)
Mixed_predict = logistic_full.predict(Mixed_f_2.T)
Tyler_score_test = logistic_full.score(Tyler_filtered_test.T, Tyler_y_test)
Henry_score_test = logistic_full.score(Henry_filtered_test.T, Henry_y_test)
Yael_score_test = logistic_full.score(Yael_filtered_test.T, Yael_y_test)
plt.figure()
plt.title('Yael Test Prediction' + '\n' + str(Yael_score_test))
plt.scatter(Yael_t_filtered_test, Yael_predict_test)
plt.ylim(-1,4)
plt.figure()
plt.title('Henry Test Prediction' + '\n' + str(Henry_score_test))
plt.scatter(Henry_t_filtered_test, Henry_predict_test)
plt.ylim(-1,4)
plt.figure()
plt.title('Tyler Test Prediction' + '\n' + str(Tyler_score_test))
plt.scatter(Tyler_t_filtered_test, Tyler_predict_test)
plt.ylim(-1,4)
plt.figure()
plt.title('Mixed Prediction')
plt.scatter(Mixed_t, Mixed_predict)
plt.ylim(-1,4)
plt.show()



