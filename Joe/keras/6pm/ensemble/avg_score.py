import pandas as pd
import numpy as np
import glob
 
# Path of data files
path = "./"
 
# List of prediction files in path
allPredsFiles = glob.glob(path+"/keras_mlp_finalresult_bag_*.txt")
 
# Array containg all the prediction
allPreds = []
 
for file in allPredsFiles:
    preds = pd.read_csv(file,header = None).values
    predictionTemplate = preds # Keep one prediction as template
    allPreds.append(preds[:,0]) #1:2 keeps dimension and removes label
 
# Stacks all the predictions (easier to treat this way)
predsStacked = (np.vstack(allPreds)).transpose()  
 
# Create the averaged result array, copying the template
predsAveraged = np.array(predictionTemplate)
 
# For each prediction, get the most frequent one
for i in range(predsStacked.shape[0]):
    (values,counts) = np.unique(predsStacked[i], return_counts=True) # Use count to find most frequent one
    ind=np.argmax(counts) # Index of the most frequent label
    predsAveraged[i] = values[ind]  # gets the most frequent label
    
# Save the averaged predicitions using Kaggle format
np.savetxt(path+"/final_avg_pred.txt", np.c_[predsAveraged], delimiter=',', fmt = '%d')