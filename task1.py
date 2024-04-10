import numpy as np
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)

#Standardizing the rows (transposing as fit_transform standardizes along columns)

#print(df)
#print(labels_df)

# Sepearate into test data and working data
n_rows = len(df) 
m_test = 900

test_data = df.head(m_test)
working_data = df.tail(n_rows-m_test)

#print(test_data)
#print(working_data)


