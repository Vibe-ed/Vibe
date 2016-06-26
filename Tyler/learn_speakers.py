from sound_to_features import wav_to_frequencies
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import  pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model



#dirs and files
#---------------------------------------------
wav_dir = '/Users/tomharamaty/Downloads'

speaker_1_abc_file = 'Yael_abc.wav'
speaker_2_abc_file = 'Henry_abc.wav'
speaker_3_abc_file = 'Tyler_abc.wav'

speaker_1_reading_file = 'Yael_ny.wav'
speaker_2_reading_file = 'Henry_ny.wav'
speaker_3_reading_file = 'Tyler_ny.wav'
#----------------------------------------------



def load_data():
	#load abc files
	speaker_1_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_abc_file))
	speaker_2_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_abc_file))
	speaker_3_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_abc_file))

	#load readings
	speaker_1_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_reading_file))
	speaker_2_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_reading_file))
	speaker_3_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_reading_file))

	return speaker_1_abc, speaker_2_abc, speaker_3_abc, speaker_1_reading, speaker_2_reading, speaker_3_reading


def load_and_arrange_data():
	"""
	loads data reading data,
	builds X matrix and y vector for learning.
	"""
	speaker_1_abc, speaker_2_abc, speaker_3_abc, speaker_1_reading, speaker_2_reading, speaker_3_reading = load_data()


	#data used for learning.
	data = (speaker_1_reading[2], speaker_2_reading[2], speaker_3_reading[2])
	labels = ([1]*len(speaker_1_reading[2]), [2]*len(speaker_2_reading[2]), [3]*len(speaker_3_reading[2]))
	X = np.concatenate(data,axis=0)
	y = np.concatenate(labels,axis=0)

	return X,y


def split_and_shuffle(X,y,test_size=0.2):

	#shuffle
	np.random.seed(0)
	indices = np.random.permutation(len(X))

	#split to train and test
	X_train, X_valid, y_train, y_valid = train_test_split(
	X[indices], y[indices], test_size=test_size)

	return X_train, X_valid, y_train, y_valid


def logistic_regression():

	"""
	runs logistic regression on data.
	disabled section - Grid search over params
	"""

	X,y = load_and_arrange_data()
	X_train, X_valid, y_train, y_valid = split_and_shuffle(X,y,test_size=0.2)


	#build scaling + estimator pipeline
	print 'building estimators'
	C=1e-2
	estimators = [('standard scaler', preprocessing.StandardScaler()), ('logistic', linear_model.LogisticRegression(C=C))]
	clf = pipeline.Pipeline(estimators)
	clf.fit(X_train,y_train)


	print 'train error: %d'%clf.score(X_train,y_train)
	print 'train error: %d'%clf.score(X_valid, y_valid)

	'''
	params = dict(logistic__C=[1])
	#cross validation gridsearch 
	print 'grid search'
	#params = dict(logistic__C=[0.1,0.3,1,3, 10,30, 100])
	grid_search = GridSearchCV(clf, param_grid=params,cv=5)
	grid_search.fit(X_train, y_train)
	print 'best param: ', grid_search.best_params_ 
	print 'best train score: ', grid_search.best_score_
	'''




logistic_regression()



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
'''