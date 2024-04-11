import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from tqdm import tqdm

scaler = preprocessing.StandardScaler()

df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)


#print(df)
#print(labels_df)

# Sepearate into test data and working data
n_rows = len(df) 
m_test = 900

test_data = df.head(m_test)
working_data = df.tail(n_rows-m_test)
scaled_work_data = pd.DataFrame(scaler.fit_transform(working_data.transpose()), columns=df.columns)

#print(test_data)
#print(working_data)
#Standardizing the rows (transposing as fit_transform standardizes along columns)?
# Use PCA and cross-validation



