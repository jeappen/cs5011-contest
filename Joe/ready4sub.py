import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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

            
train_test_split_fraction = 0.4
if 'df' not in locals():
    (df, df_ldb) = read_data_full()
    df = pd.concat([df, pd.get_dummies(df['label'], prefix='class')], axis=1)
cols = df.columns.tolist()

output_index_start = -12
label_col_index = output_index_start-1

testcols = cols#['vote','pvp']+['addedvar'+str(startnum) for startnum in startnum_list]#+cols[30:39]#+[y for x in colname_set[:] for y in x]#colname_set[0]+colname_set[1]
                             
df_train=df#, df_test = train_test_split(df[testcols],\
df_test = df_ldb#                         test_size = train_test_split_fraction,random_state =1)

train_data = df_train.values
test_data = df_test.values

#In case of potato PC
del df_train,df_test

Ylabel_train = train_data[:,label_col_index]
#Ylabel_test = test_data[:,label_col_index]
Y_train = train_data[:,output_index_start:]
Y_test = None#test_data[:,output_index_start:]
X_train = train_data[:,:label_col_index] #-1 to skip last column
X_test = test_data#[:,:label_col_index]