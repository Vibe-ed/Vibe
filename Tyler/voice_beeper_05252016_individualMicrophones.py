import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
from scipy.signal import get_window
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from numpy.linalg import norm

# Filter out any values below a certain sound threshold for training sets
def filter(matrix, times, limit):
    filtered = []
    times_2 = []
    for i in range(np.shape(matrix)[1]):
        if np.sum(matrix[:,i])>limit:
            filtered.append(matrix[:,i])
            times_2.append(times[i])
    filtered = np.array(filtered).T
    return(filtered, times_2)
    
def normalize(matrix):
    matrix_norm = matrix/norm(matrix, axis = 0)
    return(matrix_norm)
        
        

# Address of the data files        
address = r'C:\Users\tchase56\Documents\Startup\voice_data\wav\\'
address_2 = r'C:\Users\tchase56\Documents\Startup\voice_data\wav\\' 
address_save = r'C:\Users\tchase56\Documents\Startup\voice_data\spectrogram\\'

# Read in the 30 second samplings
[Fs, Tyler] = scipy.io.wavfile.read(address + 'Tyler_ind_02.wav')
[Fs, Henry] = scipy.io.wavfile.read(address + 'Henry_ind_02.wav')
[Fs, Yael] = scipy.io.wavfile.read(address + 'Yael_ind_02.wav')
# Read in the 8 second samplings
[Fs, Tyler_test] = scipy.io.wavfile.read(address + 'Tyler_ind_01.wav')
[Fs, Henry_test] = scipy.io.wavfile.read(address + 'Henry_ind_01.wav')
[Fs, Yael_test] = scipy.io.wavfile.read(address + 'Yael_ind_01.wav')
# Read in the mixed sample
[Fs, Yael_mixed] = scipy.io.wavfile.read(address + 'Yael_01_shortened.wav')
[Fs, Henry_mixed] = scipy.io.wavfile.read(address + 'Henry_01_shortened.wav')
[Fs, Tyler_mixed] = scipy.io.wavfile.read(address + 'Tyler_01_shortened.wav')

    
# Use built in spectrogram to compare to FFT
pointsFactor = 1.0
overlapFactor = 1.0/2
[Tyler_F, Tyler_t, Tyler_f_2] = spectrogram(Tyler[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Henry_F, Henry_t, Henry_f_2] = spectrogram(Henry[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Yael_F, Yael_t, Yael_f_2] = spectrogram(Yael[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Tyler_F_test, Tyler_t_test, Tyler_f_2_test] = spectrogram(Tyler_test[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Henry_F_test, Henry_t_test, Henry_f_2_test] = spectrogram(Henry_test[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Yael_F_test, Yael_t_test, Yael_f_2_test] = spectrogram(Yael_test[:,0],Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Yael_mixed_F, Yael_mixed_t, Yael_mixed_f_2] = spectrogram(Yael_mixed,Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Henry_mixed_F, Henry_mixed_t, Henry_mixed_f_2] = spectrogram(Henry_mixed,Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
[Tyler_mixed_F, Tyler_mixed_t, Tyler_mixed_f_2] = spectrogram(Tyler_mixed,Fs, window = get_window('hann', Fs*pointsFactor), nperseg = Fs*pointsFactor, noverlap = Fs*overlapFactor) 
None_f_2 = np.zeros((np.shape(Tyler_f_2)))
'''
np.savetxt(address_save + 'Tyler_ind_02_spectrogram', Tyler_f_2)
np.savetxt(address_save + 'Henry_ind_02_spectrogram', Henry_f_2)
np.savetxt(address_save + 'Yael_ind_02_spectrogram', Yael_f_2)

np.savetxt(address_save + 'Tyler_ind_01_spectrogram', Tyler_f_2_test)
np.savetxt(address_save + 'Henry_ind_01_spectrogram', Henry_f_2_test)
np.savetxt(address_save + 'Yael_ind_01_spectrogram', Yael_f_2_test)
'''
# Plot the magnitude spectrogram.
plt.figure()
plt.pcolormesh(Tyler_t, Tyler_F, Tyler_f_2).set_clim(0,0.010*10**5)
plt.xlim(Tyler_t[0],Tyler_t[-1])
plt.ylim(Tyler_F[0],Tyler_F[-1])
plt.title('Tyler')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.figure()
plt.pcolormesh(Henry_t, Henry_F, Henry_f_2).set_clim(0,0.010*10**5)
plt.xlim(Henry_t[0],Henry_t[-1])
plt.ylim(Henry_F[0],Henry_F[-1])
plt.title('Henry')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.figure()
plt.pcolormesh(Yael_t, Yael_F, Yael_f_2).set_clim(0,0.010*10**5)
plt.xlim(Yael_t[0],Yael_t[-1])
plt.ylim(Yael_F[0],Yael_F[-1])
plt.title('Yael')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar()
plt.show()


# Filter out time points with no sound
limit = 30*pointsFactor/(1/100)
[Tyler_filtered, Tyler_t_filtered] = filter(Tyler_f_2, Tyler_t, limit)
[Henry_filtered, Henry_t_filtered] = filter(Henry_f_2, Henry_t, limit)
[Yael_filtered, Yael_t_filtered] = filter(Yael_f_2, Yael_t, limit)
[Tyler_filtered_test, Tyler_t_filtered_test] = filter(Tyler_f_2_test, Tyler_t_test, limit)
[Henry_filtered_test, Henry_t_filtered_test] = filter(Henry_f_2_test, Henry_t_test, limit)
[Yael_filtered_test, Yael_t_filtered_test] = filter(Yael_f_2_test, Yael_t_test, limit)
None_t = Tyler_t
Tyler_t = Tyler_t_filtered
Henry_t = Henry_t_filtered
Yael_t = Yael_t_filtered
Tyler_f_2 = Tyler_filtered
Henry_f_2 = Henry_filtered
Yael_f_2 = Yael_filtered

'''
# Normalize the data
Tyler_filtered = normalize(Tyler_filtered)
Henry_filtered = normalize(Henry_filtered)
Yael_filtered = normalize(Yael_filtered)
Tyler_filtered_test = normalize(Tyler_filtered_test)
Henry_filtered_test = normalize(Henry_filtered_test)
Yael_filtered_test = normalize(Yael_filtered_test)
#Tyler_filtered = normalize(Tyler_filtered)
'''


# Form y data
None_y_Tyler = np.zeros(np.shape(None_f_2)[1])
Tyler_y_Tyler = np.ones(np.shape(Tyler_filtered)[1])
Henry_y_Tyler = np.zeros(np.shape(Henry_filtered)[1])
Yael_y_Tyler = np.zeros(np.shape(Yael_filtered)[1])
y_Tyler = np.matrix(np.concatenate((None_y_Tyler,Tyler_y_Tyler,Henry_y_Tyler,Yael_y_Tyler)))

None_y_Henry = np.zeros(np.shape(None_f_2)[1])
Tyler_y_Henry = np.zeros(np.shape(Tyler_filtered)[1])
Henry_y_Henry = np.ones(np.shape(Henry_filtered)[1])
Yael_y_Henry = np.zeros(np.shape(Yael_filtered)[1])
y_Henry = np.matrix(np.concatenate((None_y_Henry,Tyler_y_Henry,Henry_y_Henry,Yael_y_Henry)))

None_y_Yael = np.zeros(np.shape(None_f_2)[1])
Tyler_y_Yael = np.zeros(np.shape(Tyler_filtered)[1])
Henry_y_Yael = np.zeros(np.shape(Henry_filtered)[1])
Yael_y_Yael = np.ones(np.shape(Yael_filtered)[1])
y_Yael = np.matrix(np.concatenate((None_y_Yael,Tyler_y_Yael,Henry_y_Yael,Yael_y_Yael)))

#test
Tyler_y_test_Tyler = np.ones(np.shape(Tyler_filtered_test)[1])
Henry_y_test_Tyler = np.zeros(np.shape(Henry_filtered_test)[1])
Yael_y_test_Tyler = np.zeros(np.shape(Yael_filtered_test)[1])

Tyler_y_test_Henry = np.zeros(np.shape(Tyler_filtered_test)[1])
Henry_y_test_Henry = np.ones(np.shape(Henry_filtered_test)[1])
Yael_y_test_Henry = np.zeros(np.shape(Yael_filtered_test)[1])

Tyler_y_test_Yael = np.zeros(np.shape(Tyler_filtered_test)[1])
Henry_y_test_Yael = np.zeros(np.shape(Henry_filtered_test)[1])
Yael_y_test_Yael = np.ones(np.shape(Yael_filtered_test)[1])

# Form x data
samples = np.shape(Tyler_filtered)[1] + np.shape(Henry_filtered)[1] + np.shape(Yael_filtered)[1] + np.shape(None_f_2)[1]
variables = np.shape(Tyler_filtered)[0]
x = np.concatenate((None_f_2,Tyler_filtered,Henry_filtered,Yael_filtered), axis = 1)

# randomize x and y
temp = np.concatenate((y_Tyler, y_Henry, y_Yael ,x), axis = 0).T
np.random.seed([131])
np.random.shuffle(temp)
temp = temp.T
y_shuffled_Tyler = np.matrix((temp[0,:])).T
y_shuffled_Tyler = np.array(y_shuffled_Tyler)
y_shuffled_Tyler = y_shuffled_Tyler.reshape((samples,))

y_shuffled_Henry = np.matrix((temp[1,:])).T
y_shuffled_Henry = np.array(y_shuffled_Henry)
y_shuffled_Henry = y_shuffled_Henry.reshape((samples,))

y_shuffled_Yael = np.matrix((temp[2,:])).T
y_shuffled_Yael = np.array(y_shuffled_Yael)
y_shuffled_Yael = y_shuffled_Yael.reshape((samples,))

x_shuffled = np.matrix(temp[3:,:]).T

# Plot fit of Training
'''
logistic_full_Tyler = AdaBoostClassifier(n_estimators = 50, random_state = 131)
logistic_full_Tyler = AdaBoostClassifier(n_estimators = 50, random_state = 131)
logistic_full_Tyler = AdaBoostClassifier(n_estimators = 50, random_state = 131)
'''
'''
logistic_full_Tyler = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 0.5)
logistic_full_Henry = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 0.5)
logistic_full_Yael = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 0.5)
'''

logistic_full_Tyler = RandomForestClassifier(n_estimators = 200, max_features = 0.25)
logistic_full_Henry = RandomForestClassifier(n_estimators = 200, max_features = 0.25)
logistic_full_Yael = RandomForestClassifier(n_estimators = 200, max_features = 0.25)


logistic_full_Tyler.fit(x_shuffled, y_shuffled_Tyler)
logistic_full_Henry.fit(x_shuffled, y_shuffled_Henry)
logistic_full_Yael.fit(x_shuffled, y_shuffled_Yael)

# Training Error
Tyler_predict = logistic_full_Tyler.predict(Tyler_filtered.T)
Henry_predict = logistic_full_Tyler.predict(Henry_filtered.T)
Yael_predict = logistic_full_Tyler.predict(Yael_filtered.T)
None_predict = logistic_full_Tyler.predict(None_f_2.T)
Tyler_score = logistic_full_Tyler.score(Tyler_filtered.T, Tyler_y_Tyler)
Henry_score = logistic_full_Tyler.score(Henry_filtered.T, Henry_y_Tyler)
Yael_score = logistic_full_Tyler.score(Yael_filtered.T, Yael_y_Tyler)
None_score = logistic_full_Tyler.score(None_f_2.T, None_y_Tyler)
plt.figure()
plt.title('Yael Training Prediction for Tyler' + '\n' + str(Yael_score))
plt.scatter(Yael_t, Yael_predict)
plt.ylim(-1,2)
plt.figure()
plt.title('Henry Training Prediction for Tyler' + '\n' + str(Henry_score))
plt.scatter(Henry_t, Henry_predict)
plt.ylim(-1,2)
plt.figure()
plt.title('Tyler Training Prediction for Tyler' + '\n' + str(Tyler_score))
plt.scatter(Tyler_t, Tyler_predict)
plt.ylim(-1,2)
plt.figure()
plt.title('Silence Training Prediction for Tyler' + '\n' + str(None_score))
plt.scatter(None_t, None_predict)
plt.ylim(-1,2)
plt.show()

Tyler_predict = logistic_full_Henry.predict(Tyler_filtered.T)
Henry_predict = logistic_full_Henry.predict(Henry_filtered.T)
Yael_predict = logistic_full_Henry.predict(Yael_filtered.T)
None_predict = logistic_full_Henry.predict(None_f_2.T)
Tyler_score = logistic_full_Henry.score(Tyler_filtered.T, Tyler_y_Henry)
Henry_score = logistic_full_Henry.score(Henry_filtered.T, Henry_y_Henry)
Yael_score = logistic_full_Henry.score(Yael_filtered.T, Yael_y_Henry)
None_score = logistic_full_Henry.score(None_f_2.T, None_y_Henry)
plt.figure()
plt.title('Yael Training Prediction for Henry' + '\n' + str(Yael_score))
plt.scatter(Yael_t, Yael_predict)
plt.ylim(-1,2)
plt.figure()
plt.title('Henry Training Prediction for Henry' + '\n' + str(Henry_score))
plt.scatter(Henry_t, Henry_predict)
plt.ylim(-1,2)
plt.figure()
plt.title('Tyler Training Prediction for Henry' + '\n' + str(Tyler_score))
plt.scatter(Tyler_t, Tyler_predict)
plt.ylim(-1,2)
plt.figure()
plt.title('Silence Training Prediction for Henry' + '\n' + str(None_score))
plt.scatter(None_t, None_predict)
plt.ylim(-1,2)
plt.show()


Tyler_predict_Tyler = logistic_full_Tyler.predict(Tyler_filtered.T)
Henry_predict = logistic_full_Yael.predict(Henry_filtered.T)
Yael_predict = logistic_full_Yael.predict(Yael_filtered.T)
None_predict = logistic_full_Yael.predict(None_f_2.T)
Tyler_score = logistic_full_Yael.score(Tyler_filtered.T, Tyler_y_Yael)
Henry_score = logistic_full_Yael.score(Henry_filtered.T, Henry_y_Yael)
Yael_score = logistic_full_Yael.score(Yael_filtered.T, Yael_y_Yael)
None_score = logistic_full_Yael.score(None_f_2.T, None_y_Yael)
plt.figure()
plt.title('Yael Training Prediction for Yael' + '\n' + str(Yael_score))
plt.scatter(Yael_t, Yael_predict)
plt.ylim(-1,4)
plt.figure()
plt.title('Henry Training Prediction for Yael ' + '\n' + str(Henry_score))
plt.scatter(Henry_t, Henry_predict)
plt.ylim(-1,4)
plt.figure()
plt.title('Tyler Training Prediction for Yael' + '\n' + str(Tyler_score))
plt.scatter(Tyler_t, Tyler_predict)
plt.ylim(-1,4)
plt.figure()
plt.title('Silence Training for Yael' + '\n' + str(None_score))
plt.scatter(None_t, None_predict)
plt.ylim(-1,4)
plt.show()


# Test Error
Tyler_predict_test = logistic_full_Tyler.predict(Tyler_filtered_test.T)
Henry_predict_test = logistic_full_Tyler.predict(Henry_filtered_test.T)
Yael_predict_test = logistic_full_Tyler.predict(Yael_filtered_test.T)
Mixed_predict = logistic_full_Tyler.predict(Tyler_mixed_f_2.T)
Tyler_score_test = logistic_full_Tyler.score(Tyler_filtered_test.T, Tyler_y_test_Tyler)
Henry_score_test = logistic_full_Tyler.score(Henry_filtered_test.T, Henry_y_test_Tyler)
Yael_score_test = logistic_full_Tyler.score(Yael_filtered_test.T, Yael_y_test_Tyler)
plt.figure()
plt.title('Yael Test Prediction for Tyler' + '\n' + str(Yael_score_test))
plt.scatter(Yael_t_filtered_test, Yael_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Henry Test Prediction for Tyler' + '\n' + str(Henry_score_test))
plt.scatter(Henry_t_filtered_test, Henry_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Tyler Test Prediction for Tyler' + '\n' + str(Tyler_score_test))
plt.scatter(Tyler_t_filtered_test, Tyler_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Mixed Prediction for Tyler')
plt.scatter(Tyler_mixed_t, Mixed_predict)
plt.ylim(-1,4)
plt.show()


Tyler_predict_test = logistic_full_Henry.predict(Tyler_filtered_test.T)
Henry_predict_test = logistic_full_Henry.predict(Henry_filtered_test.T)
Yael_predict_test = logistic_full_Henry.predict(Yael_filtered_test.T)
Mixed_predict = logistic_full_Henry.predict(Henry_mixed_f_2.T)
Tyler_score_test = logistic_full_Henry.score(Tyler_filtered_test.T, Tyler_y_test_Henry)
Henry_score_test = logistic_full_Henry.score(Henry_filtered_test.T, Henry_y_test_Henry)
Yael_score_test = logistic_full_Henry.score(Yael_filtered_test.T, Yael_y_test_Henry)
plt.figure()
plt.title('Yael Test Prediction for Henry' + '\n' + str(Yael_score_test))
plt.scatter(Yael_t_filtered_test, Yael_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Henry Test Prediction for Henry' + '\n' + str(Henry_score_test))
plt.scatter(Henry_t_filtered_test, Henry_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Tyler Test Prediction for Henry' + '\n' + str(Tyler_score_test))
plt.scatter(Tyler_t_filtered_test, Tyler_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Mixed Prediction for Henry')
plt.scatter(Henry_mixed_t, Mixed_predict)
plt.ylim(-1,2)
plt.show()


Tyler_predict_test = logistic_full_Yael.predict(Tyler_filtered_test.T)
Henry_predict_test = logistic_full_Yael.predict(Henry_filtered_test.T)
Yael_predict_test = logistic_full_Yael.predict(Yael_filtered_test.T)
Mixed_predict = logistic_full_Yael.predict(Yael_mixed_f_2.T)
Tyler_score_test = logistic_full_Yael.score(Tyler_filtered_test.T, Tyler_y_test_Yael)
Henry_score_test = logistic_full_Yael.score(Henry_filtered_test.T, Henry_y_test_Yael)
Yael_score_test = logistic_full_Yael.score(Yael_filtered_test.T, Yael_y_test_Yael)
plt.figure()
plt.title('Yael Test Prediction for Yael' + '\n' + str(Yael_score_test))
plt.scatter(Yael_t_filtered_test, Yael_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Henry Test Prediction for Yael' + '\n' + str(Henry_score_test))
plt.scatter(Henry_t_filtered_test, Henry_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Tyler Test Prediction for Yael' + '\n' + str(Tyler_score_test))
plt.scatter(Tyler_t_filtered_test, Tyler_predict_test)
plt.ylim(-1,2)
plt.figure()
plt.title('Mixed Prediction for Yael')
plt.scatter(Yael_mixed_t, Mixed_predict)
plt.ylim(-1,2)
plt.show()
