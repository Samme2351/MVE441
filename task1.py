import numpy as np
import pandas as pd

df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)

#Standardizing the rows (transposing as fit_transform standardizes along columns)

print(df)
print(labels_df)

