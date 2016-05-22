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
        if np.sum(matrix[:,i])>30:
            filtered.append(matrix[:,i])
            times_2.append(times[i])
    filtered = np.array(filtered).T
    return(filtered, times_2)

# Address of the data files       
address = r'C:\Users\tchase56\Documents\Startup\voice_data\\'
address_2 = r'C:\Users\tchase56\Documents\Startup\voice_data\\'

# Read in the 10 second samplings
[Fs, Tyler] = scipy.io.wavfile.read(address + 'Tyler_in.wav')
[Fs, Henry] = scipy.io.wavfile.read(address + 'Henry_in.wav')
[Fs, Yael] = scipy.io.wavfile.read(address + 'Yael_in.wav')
    
Fs = Fs        

# Use built in spectrogram to compare to FFT
pointsFactor = 1.0/100
[Tyler_F, Tyler_t, Tyler_f_2] = spectrogram(Tyler[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Henry_F, Henry_t, Henry_f_2] = spectrogram(Henry[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
[Yael_F, Yael_t, Yael_f_2] = spectrogram(Yael[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor) 
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

# Form x data
samples = np.shape(Tyler_f_2)[1] + np.shape(Henry_f_2)[1] + np.shape(Yael_f_2)[1] + np.shape(None_f_2)[1]
variables = np.shape(Tyler_f_2)[0]
x = np.concatenate((None_f_2,Tyler_f_2,Henry_f_2,Yael_f_2), axis = 1)

# randomize x and y
temp = np.concatenate((y,x), axis = 0).T
np.random.shuffle(temp)
temp = temp.T
y_shuffled = np.matrix((temp[0,:])).T
y_shuffled = np.array(y_shuffled)
y_shuffled = y_shuffled.reshape((samples,))
x_shuffled = np.matrix(temp[1:,:]).T

# Create a training and a test set (hold out 30% for testing)
cutOff = int(np.shape(x_shuffled)[0]*0.3)
x_test = x_shuffled[:cutOff,:]
y_test = y_shuffled[:cutOff]
x_train = x_shuffled[cutOff:,:]
y_train = y_shuffled[cutOff:]

# Logistic Regression
logistic = LogisticRegression(penalty = 'l1', solver= 'liblinear')
logistic.fit(x_train,y_train)
accuracy_training = logistic.score(x_train,y_train)
accuracy_test = logistic.score(x_test,y_test)
print("training accuracy is " + str(accuracy_training))
print("testing accuracy is " + str(accuracy_test))

# Plot fit of Training
logistic_full = LogisticRegression(penalty = 'l1', solver = 'liblinear')
logistic_full.fit(x_shuffled, y_shuffled)
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


