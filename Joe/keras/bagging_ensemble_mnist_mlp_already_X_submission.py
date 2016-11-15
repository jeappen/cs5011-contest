'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import load_model
from sklearn.ensemble import BaggingClassifier

model_name = 'attempt2-keras-bag-mlp-mnist-50epo'
load_old_model = False
save_model = True
model_save_loc = './models/'

def read_data_full():
    f_train = open('../DATASET/train_data', 'r')
    f_labels = open('../DATASET/train_labels', 'r')
    f_ldb = open('../DATASET/leaderboardTest_data', 'r')
    f_final = open('../test_data', 'r')
#    print z.namelist()
    df_train = pd.read_csv(f_train,header = None,dtype=np.float32)
    df_train['label'] = pd.read_csv(f_labels,header = None,dtype=np.uint8)
    df_ldb = pd.read_csv(f_ldb, header = None,dtype=np.float32)
    df_final = pd.read_csv(f_final, header = None,dtype=np.float32)
    return df_train,df_ldb,df_final

if 'df' not in locals():
    (df, df_ldb, df_final) = read_data_full()
    df = pd.concat([df, pd.get_dummies(df['label'], prefix='class')], axis=1)
cols = df.columns.tolist()

output_index_start = -12
label_col_index = output_index_start-1    

testcols = cols

train_data = df.values
test_data = df_ldb.values
final_test_data = df_final.values


Ylabel_train = train_data[:,label_col_index]
Ylabel_test = test_data[:,label_col_index]
y_train = np.array(Ylabel_train.astype(np.uint8))[:,np.newaxis]     #for keras     
y_test = None#np.array(Ylabel_test.astype(np.uint8))[:,np.newaxis]  

Y_train = train_data[:,output_index_start:]
Y_test = None
X_train = train_data[:,:label_col_index] #-1 to skip last column
X_test = test_data


batch_size = 64
nb_classes = 12
nb_epoch = 55

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

num_models = 18
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
num_feat = 3072
n_hidden = int(num_feat/1.53125)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)


N = Y_train.shape[0]

bagging_index_set = []

i=0
a=[]
bag_size = N
while (len(a)!=N):
	s = np.random.choice(N,bag_size)
	bagging_index_set = bagging_index_set +[s]
	a = np.unique(np.hstack((a,s)))
	i +=1
print(i)

num_models = len(bagging_index_set)
# i=0
# a=[]
# while (len(a)!=N):
# 	s = np.random.choice(N,bag_size)
# 	bagging_index_set = bagging_index_set +[s]
# 	a = np.unique(np.hstack((a,s)))
# 	i +=1
# print i


runID = 1
for runID in range (num_models):   
	X_bag = np.array([X_train[i,:] for i in bagging_index_set[runID]])
	Y_bag = np.array([Y_train[i,:] for i in bagging_index_set[runID]])
	if not load_old_model:
		model = Sequential()
		model.add(Dense(n_hidden, input_shape=(num_feat,)))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(n_hidden))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
		model.add(Dense(12))
		model.add(Activation('softmax'))

		model.summary()

		sgd = SGD(lr=0.003, decay=1e-6, momentum=0.09, nesterov=True)
		adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		rms= RMSprop()

		model.compile(loss='categorical_crossentropy',
		              optimizer=adam,
		              metrics=['accuracy'])
		# bagging = BaggingClassifier(model,max_samples=0.5, max_features=1.0)
		history = model.fit(X_bag, Y_bag,
		                    batch_size=batch_size, nb_epoch=nb_epoch,
		                    verbose=1)
		if save_model:
			print('Saving model')
			model.save(model_save_loc+model_name+str(runID)+'.h5')
	else:
		print('loading old model')
		model = load_model(model_name+'.h5')
		model.summary()

	Predictions = model.predict_classes(X_test)

	print(Predictions)
	result = np.c_[Predictions]
	df_result = pd.DataFrame(result)

	Predictions = model.predict_classes(final_test_data.astype(np.float32))
	result = np.c_[Predictions]
	df_final_result = pd.DataFrame(result)


	df_result.to_csv('../../results/ensemble/bagging/keras_mlp_result_bag_ens'+str(runID)+'.txt', index=False, header = None)
	df_final_result.to_csv('../../results/ensemble/bagging/final/keras_mlp_finalresult_bag_ens'+str(runID)+'.txt', index=False, header = None)# score = model.evaluate(X_test, Y_test, verbose=0)

# print('Test score:', score[0]
# print('Test accuracy:', score[1])