import pandas as pd
import zipfile
z = zipfile.ZipFile('..\DATASET.zip')
print z.namelist()
df_train = pd.read_csv(z.open(z.namelist()[3]),header = None)
df_train['label'] = pd.read_csv(z.open(z.namelist()[4]),header = None)
df_ldb = pd.read_csv(z.open(z.namelist()[1]), header = None)
