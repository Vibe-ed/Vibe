from sound_to_features import wav_to_frequencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import  pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import defaultdict
import tensorflow as tf

#dirs and files
#---------------------------------------------
wav_dir = '/Users/tomharamaty/Downloads'

speaker_1_abc_file = 'Yael_abc.wav'
speaker_2_abc_file = 'Henry_abc.wav'
speaker_3_abc_file = 'Tyler_abc.wav'

speaker_1_reading_file = 'Yael_ny.wav'
speaker_2_reading_file = 'Henry_ny.wav'
speaker_3_reading_file = 'Tyler_ny.wav'

speaker_1_girl_file = 'Yael_girl.wav'
speaker_2_girl_file = 'Henry_girl.wav'
speaker_3_girl_file = 'Tyler_girl.wav'

group_file = 'group_2.wav'
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

def load_data_abc_in_training():

	#load abc files
	speaker_1_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_abc_file))
	speaker_2_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_abc_file))
	speaker_3_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_abc_file))
	#load readings
	speaker_1_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_reading_file))
	speaker_2_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_reading_file))
	speaker_3_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_reading_file))

	#load girl story
	speaker_1_girl = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_girl_file))
	speaker_2_girl = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_girl_file))
	speaker_3_girl = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_girl_file))

	data_reading = 	(speaker_1_reading[2], speaker_2_reading[2], speaker_3_reading[2])
	labels_reading = ([0]*len(speaker_1_reading[2]), [1]*len(speaker_2_reading[2]), [2]*len(speaker_3_reading[2]))
	X_reading = np.concatenate(data_reading,axis=0)
	y_reading = np.concatenate(labels_reading,axis=0)

	data_girl = 	(speaker_1_girl[2], speaker_2_girl[2], speaker_3_girl[2])
	labels_girl = ([0]*len(speaker_1_girl[2]), [1]*len(speaker_2_girl[2]), [2]*len(speaker_3_girl[2]))
	X_girl = np.concatenate(data_girl,axis=0)
	y_girl = np.concatenate(labels_girl,axis=0)

	X_all_reading = np.concatenate((X_reading, X_girl),axis=0)
	y_all_reading = np.concatenate((y_reading, y_girl),axis=0)

	data_abc = (speaker_1_abc[2], speaker_2_abc[2], speaker_3_abc[2])
	labels_abc = ([0]*len(speaker_1_abc[2]), [1]*len(speaker_2_abc[2]), [2]*len(speaker_3_abc[2]))
	
	X_abc = np.concatenate(data_abc,axis=0)
	y_abc= np.concatenate(labels_abc,axis=0)

	#shuffle
	np.random.seed(0)
	indices = np.random.permutation(len(X_all_reading))
	#split to train and test
	X_train, X_valid, y_train, y_valid = train_test_split(
	X_all_reading[indices], y_all_reading[indices], test_size=0.2)

	X_train = np.concatenate((X_train,X_abc),axis=0)
	y_train = np.concatenate((y_train,y_abc),axis=0)

	return X_train, X_valid, y_train, y_valid

def load_all_train_data():

	#load abc files
	speaker_1_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_abc_file))
	speaker_2_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_abc_file))
	speaker_3_abc = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_abc_file))
	#load readings
	speaker_1_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_reading_file))
	speaker_2_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_reading_file))
	speaker_3_reading = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_reading_file))

	#load girl story
	speaker_1_girl = wav_to_frequencies('%s/%s'%(wav_dir, speaker_1_girl_file))
	speaker_2_girl = wav_to_frequencies('%s/%s'%(wav_dir, speaker_2_girl_file))
	speaker_3_girl = wav_to_frequencies('%s/%s'%(wav_dir, speaker_3_girl_file))

	data_reading = 	(speaker_1_reading[2], speaker_2_reading[2], speaker_3_reading[2])
	labels_reading = ([0]*len(speaker_1_reading[2]), [1]*len(speaker_2_reading[2]), [2]*len(speaker_3_reading[2]))
	X_reading = np.concatenate(data_reading,axis=0)
	y_reading = np.concatenate(labels_reading,axis=0)

	data_girl = 	(speaker_1_girl[2], speaker_2_girl[2], speaker_3_girl[2])
	labels_girl = ([0]*len(speaker_1_girl[2]), [1]*len(speaker_2_girl[2]), [2]*len(speaker_3_girl[2]))
	X_girl = np.concatenate(data_girl,axis=0)
	y_girl = np.concatenate(labels_girl,axis=0)

	data_abc = (speaker_1_abc[2], speaker_2_abc[2], speaker_3_abc[2])
	labels_abc = ([0]*len(speaker_1_abc[2]), [1]*len(speaker_2_abc[2]), [2]*len(speaker_3_abc[2]))
	X_abc = np.concatenate(data_abc,axis=0)
	y_abc= np.concatenate(labels_abc,axis=0)


	X_all_data = np.concatenate((X_reading, X_girl, X_abc),axis=0)
	y_all_data = np.concatenate((y_reading, y_girl, y_abc),axis=0)

	#shuffle
	np.random.seed(0)
	indices = np.random.permutation(len(X_all_data))

	return X_all_data[indices], y_all_data[indices]



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
<<<<<<< Updated upstream

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
=======

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

>>>>>>> Stashed changes

	#X,y = load_and_arrange_data()
	#X_train, X_valid, y_train, y_valid = split_and_shuffle(X,y,test_size=0.2)
	X_train, X_valid, y_train, y_valid = load_data_abc_in_training()

	print X_train.shape
	print X_valid.shape
	#build scaling + estimator pipeline
	print 'building estimators'
<<<<<<< Updated upstream
	C=1.5e-4
	estimators = [
					('standard scaler', preprocessing.StandardScaler()),
					('logistic', linear_model.LogisticRegression(C=C))
				]
=======
	C=1e-2
	estimators = [('standard scaler', preprocessing.StandardScaler()), ('logistic', linear_model.LogisticRegression(C=C))]
>>>>>>> Stashed changes
	clf = pipeline.Pipeline(estimators)
	clf.fit(X_train,y_train)


	print 'logistic train score: %f'%clf.score(X_train,y_train)
	print 'logistic valid score: %f'%clf.score(X_valid, y_valid)

	'''
	params = dict(logistic__C=[1])
	#cross validation gridsearch 
	print 'grid search'
	params = dict(logistic__C=[0.1,0.3,1,3, 10,30, 100])
	grid_search = GridSearchCV(clf, param_grid=params,cv=5)
	grid_search.fit(X_train, y_train)
	print 'best param: ', grid_search.best_params_ 
	print 'best train score: ', grid_search.best_score_
	'''

def svm():
	"""
	runs logistic regression on data.
	disabled section - Grid search over params
	"""

	#X,y = load_and_arrange_data()
	#X_train, X_valid, y_train, y_valid = split_and_shuffle(X,y,test_size=0.2)
	X_train, X_valid, y_train, y_valid = load_data_abc_in_training()

	#build scaling + estimator pipeline
	print 'building estimators'

	C=3
	estimators = [
					('standard scaler', preprocessing.StandardScaler()),
					('svm', SVC(kernel='rbf', C=C))
				]
	clf = pipeline.Pipeline(estimators)
	clf.fit(X_train,y_train)
	print 'svm train score: %f'%clf.score(X_train,y_train)
	print 'svm valid score: %f'%clf.score(X_valid, y_valid)
	#Best so far: 0.828 for validation with frequency features and C=3 (didn't optimize aggresively).

def random_forest():
	"""
	runs logistic regression on data.
	disabled section - Grid search over params
	"""

	#X,y = load_and_arrange_data()
	#X_train, X_valid, y_train, y_valid = split_and_shuffle(X,y,test_size=0.2)
	X_train, X_valid, y_train, y_valid = load_data_abc_in_training()
	print X_train.shape
	print X_valid.shape


	#build scaling + estimator pipeline
	print 'building estimators'

	for i in [0.004]:
		print 'n_estimators = %f'%i
		
		#buils pipeline
		estimators = [
					('standard scaler', preprocessing.StandardScaler()),
					('random_forest', RandomForestClassifier(n_estimators = 450, max_features = i ))
				]
		clf = pipeline.Pipeline(estimators)

		#fit training data.
		clf.fit(X_train, y_train)

		#print scores
		print 'random forest train score: %f'%clf.score(X_train,y_train)
		print 'random forest valid score: %f'%clf.score(X_valid, y_valid)

		#load data for plotting
		speaker_1_abc, speaker_2_abc, speaker_3_abc, speaker_1_reading, speaker_2_reading, speaker_3_reading = load_data()
		
		#predict and plot solo readings
		'''
		fig = plt.figure()
		plt.title('Yael')
		fig.add_subplot(131)
		yael = clf.predict(speaker_1_reading[2])
		yael_boosted = boost_by_sequence(yael)
		plt.plot(yael,color='b')
		plt.plot(yael_boosted,color='r')

		fig.add_subplot(132)
		plt.title('Henry')
		henry = clf.predict(speaker_2_reading[2])
		henry_boosted = boost_by_sequence(henry)
		plt.plot(henry,color='b')
		plt.plot(henry_boosted,color='r')

		fig.add_subplot(133)
		plt.title('Tyler')
		tyler = clf.predict(speaker_3_reading[2])
		tyler_boosted = boost_by_sequence(tyler)
		plt.plot(tyler,color='b')
		plt.plot(tyler_boosted,color='r')
		'''

		#predict and plot combined file.
		fig2 = plt.figure()
		group = wav_to_frequencies('%s/%s'%(wav_dir, group_file))
		group_predictions = clf.predict(group[2])
		ax = fig2.add_subplot(111)
		ax.plot(group_predictions,color='b')
		#ax.plot(boost_by_sequence(group_predictions),color='r')
		plt.show()


def fully_connected_nn():
	X_train, X_valid, y_train, y_valid = load_data_abc_in_training()

	num_of_features = X_train.shape[1]

	num_labels = 3
	valid_labels = (np.arange(num_labels) == y_valid[:,None]).astype(np.float32)
	train_labels = (np.arange(num_labels) == y_train[:,None]).astype(np.float32)

	#L2 neural network

	C_vals=[1e-4,3e-4,1e-3]
	#C_vals=[1e-3]

	for C in C_vals:
	    print('C='+str(C))
	    batch_size = 128
	    hidden_size = 1024

	    def predict(input_set,input_weights,input_biases,hidden_weights,hidden_biases):
	        hid = tf.nn.relu(tf.matmul(input_set, input_weights) + input_biases)
	        log1 = tf.matmul(hid, hidden_weights) + hidden_biases
	        return tf.nn.softmax(log1)

	    graph = tf.Graph()
	    with graph.as_default():
			# Input data. For the training data, we use a placeholder that will be fed
			# at run time with a training minibatch.
			tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, num_of_features))
			tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
			
			tf_train_dataset = tf.constant(X_train)
			tf_valid_dataset = tf.constant(X_valid)

			# Variables, first layer
			input_weights = tf.Variable(tf.truncated_normal([num_of_features, hidden_size]))
			input_biases = tf.Variable(tf.zeros([hidden_size]))

			# Variables, hidden layer
			hidden_weights = tf.Variable(
			tf.truncated_normal([hidden_size,num_labels]))
			hidden_biases = tf.Variable(tf.zeros([num_labels]))

			# Training computation.
			hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, input_weights) + input_biases)
			logits = tf.matmul(hidden1, hidden_weights) + hidden_biases
			loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))+C*(
			tf.nn.l2_loss(input_weights)+
			tf.nn.l2_loss(hidden_weights))

			# Optimizer.
			optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

			# Predictions for the training, validation, and test data.
			train_prediction = tf.nn.softmax(logits)
			valid_prediction = predict(tf_valid_dataset,input_weights,input_biases,hidden_weights,hidden_biases)
			#test_prediction = predict(tf_test_dataset,input_weights,input_biases,hidden_weights,hidden_biases)


			num_steps = 3001


			with tf.Session(graph=graph) as session:
				tf.initialize_all_variables().run()
				print("Initialized")
				for step in range(num_steps):
					# Pick an offset within the training data, which has been randomized.
					# Note: we could use better randomization across epochs.
					offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
					# Generate a minibatch.
					batch_data = X_train[offset:(offset + batch_size), :]
					batch_labels = train_labels[offset:(offset + batch_size), :]
					# Prepare a dictionary telling the session where to feed the minibatch.
					# The key of the dictionary is the placeholder node of the graph to be fed,
					# and the value is the numpy array to feed to it.
					feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
					_, l, predictions = session.run(
					  [optimizer, loss, train_prediction], feed_dict=feed_dict)
					if (step % 500 == 0):
						print("Minibatch loss at step %d: %f" % (step, l))
						print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
						print("Validation accuracy: %.1f%%" % accuracy(
						valid_prediction.eval(), valid_labels))

def boost_by_sequence(predictions,distance=1):
	boosted = []
	for i in xrange(len(predictions)):
		counts = defaultdict(int)
		for j in xrange(i-distance,i+distance+1):
			if j<0 or j>=len(predictions):
				continue
			prediction = predictions[j]
			counts[prediction] = counts[prediction]+1
		max_count = 0
		chosen_pred = -100
		for pred,count in counts.iteritems():
			if count > max_count:
				max_count = count
				chosen_pred = pred
		boosted.append(chosen_pred)
	return boosted


#fully_connected_nn()
logistic_regression()
svm()
random_forest()








