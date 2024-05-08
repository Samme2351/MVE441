import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.model_selection import train_test_split,cross_val_predict

df = pd.read_csv('./Data/CATSnDOGS.csv', sep="," ,header = 0)
labels_df = pd.read_csv('./Data/Labels.csv', header = 0)

plt.imshow(df.iloc[103].to_numpy().reshape(64,64).transpose(), cmap=colormaps['cool'])
plt.show()