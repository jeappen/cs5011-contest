import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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



# X_train_min = np.min(X_train)


# X_train =X_train- np.min(X_train)
# X_train_scale = 256/np.max(X_train) 
# X_train = X_train*X_train_scale  
# # X_train_int = X_train.astype(np.uint8)    

# X_test  = X_test - X_train_min
# X_test = X_test*X_train_scale
# X_test_int = X_test.astype(np.uint8) 
# 
# X_train = np.reshape(X_train,(X_train.shape[0],32,32,3))  

# X_test = np.reshape(X_test,(X_test.shape[0],32,32,3))  

       
