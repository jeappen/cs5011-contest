"""This is an example of using Hierarchical RNN (HRNN) to classify MNIST digits.
HRNNs can learn across multiple levels of temporal hiearchy over a complex sequence.
Usually, the first recurrent layer of an HRNN encodes a sentence (e.g. of word vectors)
into a  sentence vector. The second recurrent layer then encodes a sequence of
such vectors (encoded by the first layer) into a document vector. This
document vector is considered to preserve both the word-level and
sentence-level structure of the context.
# References
    - [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://web.stanford.edu/~jurafsky/pubs/P15-1107.pdf)
        Encodes paragraphs and documents with HRNN.
        Results have shown that HRNN outperforms standard
        RNNs and may play some role in more sophisticated generation tasks like
        summarization or question answering.
    - [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
        Achieved state-of-the-art results on skeleton based action recognition with 3 levels
        of bidirectional HRNN combined with fully connected layers.
In the below MNIST example the first LSTM layer first encodes every
column of pixels of shape (28, 1) to a column vector of shape (128,). The second LSTM
layer encodes then these 28 column vectors of shape (28, 128) to a image vector
representing the whole image. A final Dense layer is added for prediction.
After 5 epochs: train acc: 0.9858, val acc: 0.9864
"""
from __future__ import print_function

import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils


def read_data_full():
    f_train = open('/media/yokian/Storage/ml/DATASET/train_data', 'r')
    f_labels = open('/media/yokian/Storage/ml/DATASET/train_labels', 'r')
    f_ldb = open('/media/yokian/Storage/ml/DATASET/leaderboardTest_data', 'r')
#    print z.namelist()
    df_train = pd.read_csv(f_train,header = None,dtype=np.float32)
    df_train['label'] = pd.read_csv(f_labels,header = None)
    df_ldb = pd.read_csv(f_ldb, header = None,dtype=np.float32)
    return df_train,df_ldb
    

def batchify(lst,n):
    return [ lst[i::n] for i in xrange(n) ]

            
train_test_split_fraction = 0.3
parameter_grid = {
    'max_features': [0.5, 1.],
    'max_depth': [5., None]
}

#grid_search = GridSearchCV(RandomForestClassifier(n_estimators = 100), parameter_grid,
#                            cv=5, verbose=3)

#df = pd.read_csv('../Training_Dataset.csv')

if 'df' not in locals():
    (df, df_ldb) = read_data_full()
    df = pd.concat([df, pd.get_dummies(df['label'], prefix='class')], axis=1)
cols = df.columns.tolist()

output_index_start = -12
label_col_index = output_index_start-1
#bring vote to start
#cols = cols[-5:]+cols[:-5]

#df = df[cols]

#Test how correlated, sets of features are with final vote
#startnum_list = [1,6,11,16,21]
#acc_argmax = []
#colname_set = []
#for startnum in startnum_list:
#    colnames = ['mvar'+str(i+startnum) for i in xrange (len(party_dict))]
#    colname_set = colname_set + [colnames]
#    cols_party_map = dict(zip(colnames,party_names))
#    argmax_predict = df[colnames].idxmax(axis=1).map(cols_party_map).map(party_dict)
#    acc_argmax = acc_argmax+[np.sum(df[cols[0]] == argmax_predict)/(len(df)+0.0)]
#    df['addedvar'+str(startnum)] = argmax_predict

       
testcols = cols#['vote','pvp']+['addedvar'+str(startnum) for startnum in startnum_list]#+cols[30:39]#+[y for x in colname_set[:] for y in x]#colname_set[0]+colname_set[1]
                             
df_train, df_test = train_test_split(df[testcols],\
                                     test_size = train_test_split_fraction,random_state =1)

train_data = df_train.values
test_data = df_test.values

#In case of potato PC
del df_train,df_test

Ylabel_train = train_data[:,label_col_index]
Ylabel_test = test_data[:,label_col_index]
y_train = np.array(Ylabel_train.astype(np.uint8))[:,np.newaxis]     #for keras     
y_test = np.array(Ylabel_test.astype(np.uint8))[:,np.newaxis]  



Y_train = train_data[:,output_index_start:]
Y_test = test_data[:,output_index_start:]
X_train = train_data[:,:label_col_index] #-1 to skip last column
X_test = test_data[:,:label_col_index]

pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)  
print(pca.explained_variance_ratio_) 
T_test = pca.transform(X_test)
T_train = pca.transform(X_train)

indice_set = 32**2

X_train = T_train[:,:indice_set]

X_test = T_test[:,:indice_set]

# Training parameters.
batch_size = 32
nb_classes = 12
nb_epochs = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

# The data, shuffled and split between train and test sets.
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshapes data to 4D for Hierarchical RNN.
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Converts class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

row, col, pixel = X_train.shape[1:]

# 4D input.
x = Input(shape=(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(output_dim=row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Final predictions and model.
prediction = Dense(nb_classes, activation='softmax')(encoded_columns)
model = Model(input=x, output=prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Training.
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, Y_test))

# Evaluation.
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])