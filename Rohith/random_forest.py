from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import cross_val_score

#EDIT BY JOE

execfile('../Joe/read_data.py')
df = df_train #pd.read_pickle('train_data_pickle')

#CLOSE JOE EDIT
features = range(3072)
ACCURACY = []
for i in [1,10,100,250,500,750,1000,2000]:
	print i
	model = RandomForestClassifier(n_estimators = i)
	accuracy = cross_val_score(model, df[features], df['label'], cv=5, scoring='accuracy')
	ACCURACY.append(accuracy)
plt.plot(ACCURACY)
plt.show()
	