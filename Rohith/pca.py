from sklearn.decomposition import PCA
import pylab
import pandas as pd
import numpy as np

#EDIT BY JOE

execfile('../Joe/read_data.py')
df = df_train #pd.read_pickle('train_data_pickle')

#CLOSE JOE EDIT

features = range(3072)

pca = PCA(n_components=3)
pca.fit(df[features])
pca_values = pca.transform(df[features][df['label']==1])

pylab.scatter(pca_values[:,0],pca_values[:,1],pca_values[:,2],c=df['label'][df['label']==1].values)
pylab.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_values[:,0],pca_values[:,1],pca_values[:,2], zdir='z', s=20, c=df['label'][df['label']==1].values, depthshade=True)
plt.show()
