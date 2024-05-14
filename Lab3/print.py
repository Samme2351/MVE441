import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.model_selection import train_test_split,cross_val_predict

animal = ["cat", "dog"]
df = pd.read_csv('./Data/CATSnDOGS.csv', sep="," ,header = 0)
labels_df = pd.read_csv('./Data/Labels.csv', header = 0)

data_knn = pd.read_csv('./data_LR', sep =" ", header= 0, index_col=0)
data_LR = pd.read_csv('./data_knn', sep =" ", header= 0, index_col=0)


def plot_ind(daf):
    for i in daf.columns:
        plt.imshow(df.iloc[int(i)].to_numpy().reshape(64,64).transpose(), cmap=colormaps['bone'])
        plt.title(f"This is {animal[labels_df.at[int(i),'x']]} but we believe it to be the other {daf.at[0,i]} times")
        plt.show()

pictures = 0
com_errs = 0

'''
for i in range(199):
    plt.imshow(df.iloc[i].to_numpy().reshape(64,64).transpose(), cmap=colormaps['bone'])
    plt.title(f"This is picture {i+2} and is {animal[labels_df.at[int(i),'x']]}")
    plt.show()
'''

#Show pictures
if pictures == 1:
    plot_ind(data_knn)
    plot_ind(data_LR)
    

#Common errors
if com_errs == 1 :
    com_errors=dict()
    LR_errors = []
    knn_errors = []

    for index in data_knn:
        if index in data_LR:
            com_errors[index] = (data_knn[index].iloc[0], data_LR[index].iloc[0])
        else:
            knn_errors.append(index)

    for index in data_LR:
        if index not in data_knn:
            LR_errors.append(index)

    print(com_errors)
    print(LR_errors)
    print(knn_errors)




