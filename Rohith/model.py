import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def pickle_train_data():
	df = pd.read_csv('train_data',header=None)
	df_labels = pd.read_csv('train_labels',header=None)
	df['label'] = df_labels
	df.to_pickle('train_data_pickle')

if __name__ == '__main__':
	"""
	EDIT BY JOE
	"""
	execfile('../Joe/read_data.py')
	#pickle_train_data()
	df = df_train #pd.read_pickle('train_data_pickle')
	"""
	CLOSE JOE EDIT
	"""
	features = range(3072)
	#df=df[:1000]
	'''df = df[:500]
	MODELS =  [ RandomForestClassifier(),
				SVC(),
				NuSVC(),
				
				DecisionTreeClassifier(),
				AdaBoostClassifier(),
				GradientBoostingClassifier(),
				LinearSVC(),
				GaussianNB(),
    			QuadraticDiscriminantAnalysis(),
    			KNeighborsClassifier(3)
				]

	MODELS_INDEX = ['Random Forest',
					'SVC',
					'NuSVC',
					
					'Decision Tree',
					'Adaboost',
					'Gradient Boost',
					'LinearSVC',
					'Naive Bayes',
					'QDA',
					'Nearest Neighbors'
					]'''

	MODEL_ACCURACY = {}
	model = KMeans(n_clusters=2)
	model.fit(df[features])
	print Counter(model.labels_)
	
	for i,model in enumerate(MODELS):
	
		accuracy = cross_val_score(model, df[features], df['label'], cv=10, scoring='accuracy')
		MODEL_ACCURACY[MODELS_INDEX[i]] = accuracy
		print MODELS_INDEX[i] + ' : ' +str(np.mean(accuracy))
		print accuracy
		pickle.dump(MODEL_ACCURACY,open('MODEL_ACCURACY','wb'))

	df_results = pd.DataFrame(MODEL_ACCURACY,index=MODELS_INDEX)
	df_results.to_csv('Results.csv')

