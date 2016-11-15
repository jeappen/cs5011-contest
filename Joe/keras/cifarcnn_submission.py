from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.utils import np_utils

batch_size = 100
nb_classes = 12
nb_epoch = 400
data_augmentation = True

model_name = 'keras-cnn-cifar-400-data-aug'
save_model = True
load_old_model = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
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
# y_test = np.array(Ylabel_test.astype(np.uint8))[:,np.newaxis]  

Y_train = train_data[:,output_index_start:]
# Y_test = test_data[:,output_index_start:]
X_train = train_data[:,:label_col_index] #-1 to skip last column
X_test = test_data



# X_train_min = np.min(X_train)


# X_train =X_train- np.min(X_train)
# X_train_scale = 256/np.max(X_train) 
# X_train = X_train*X_train_scale  
# # X_train_int = X_train.astype(np.uint8)    

# X_test  = X_test - X_train_min
# X_test = X_test*X_train_scale
# X_test_int = X_test.astype(np.uint8) 
# 
X_train = np.reshape(X_train,(X_train.shape[0],32,32,3))  

X_test = np.reshape(X_test,(X_test.shape[0],32,32,3))  



print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

if not load_old_model:
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.09, nesterov=True)
    adam = Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    # X_train /= 255
    # X_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            #modifying feat wise center and feat normalisation
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch)
    if save_model:
        print('Saving model')
        model.save(model_name+'.h5')
else:
    print('loading old model')
    model = load_model(model_name+'.h5')
    model.summary()


Predictions = model.predict_classes(X_test)

print(Predictions)
result = np.c_[Predictions]
df_result = pd.DataFrame(result)
df_result.to_csv('../../results/'+model_name+'.txt', index=False, header = None)

Predictions = model.predict_classes(final_test_data.astype(np.float32))
result = np.c_[Predictions]
df_final_result = pd.DataFrame(result)
df_final_result.to_csv('../../results/'+model_name+'_final.txt', index=False, header = None)# score = model.evaluate(X_test, Y_test, verbose=0)

#FOR MAHESH's PC


os.system('../../gdrive upload ../../results/'+model_name'.txt')
os.system('../../gdrive upload ../../results/'+model_name+'_final.txt')
# os.system('shutdown -t 5')
# os.system('notify-send \"PC GOING TO SHUTDOWN IN 5 MIN. type shutdown -c to cancel\" ')
# os.system('notify-send \"I REPEAT PC GOING TO SHUTDOWN IN 5 MIN. type shutdown -c to cancel\" ')