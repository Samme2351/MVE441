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

acc = pd.read_csv('./data_acc', sep=" ", header=0, index_col=0)

data_knn = pd.read_csv('./data_LR', sep =" ", header= 0, index_col=0)
data_mat_knn = pd.read_csv('./data_mat_knn', sep =" ", header= 0, index_col=0)
data_LR = pd.read_csv('./data_knn', sep =" ", header= 0, index_col=0)
data_mat_LR = pd.read_csv('./data_mat_LR', sep =" ", header= 0, index_col=0)
data_svc= pd.read_csv('./data_svc', sep =" ", header= 0, index_col=0)
data_mat_svc = pd.read_csv('./data_mat_svc', sep =" ", header= 0, index_col=0)
data_XGB = pd.read_csv('./data_XGB', sep =" ", header= 0, index_col=0)
data_mat_XGB = pd.read_csv('./data_mat_XGB', sep =" ", header= 0, index_col=0)
data_LDA = pd.read_csv('./data_LDA', sep =" ", header= 0, index_col=0)
data_mat_LDA = pd.read_csv('./data_mat_LDA', sep =" ", header= 0, index_col=0)
data_nn = pd.read_csv('./data_nn', sep =" ", header= 0, index_col=0)
data_mat_nn = pd.read_csv('./data_mat_nn', sep =" ", header= 0, index_col=0)

def plot_ind(daf):
    for i in daf.columns:
        plt.imshow(df.iloc[int(i)].to_numpy().reshape(64,64).transpose(), cmap=colormaps['bone'])
        plt.title(f"This is {animal[labels_df.at[int(i),'x']]} but we believe it to be the other {daf.at[0,i]} times")
        plt.show()

def reverse_sort(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))


pictures = 0
com_errs = 0
accuracy = 1
class_err = 0
err20 = 1

iter = 10

if err20 == 1:
    data_knn = data_knn[data_knn>=0.2*iter]
    data_knn.dropna(axis=1, inplace=True)
    data_LR = data_LR[data_knn>=0.2*iter]
    data_LR.dropna(axis=1, inplace=True)
    data_svc = data_svc[data_knn>=0.2*iter]
    data_svc.dropna(axis=1, inplace=True)
    data_XGB = data_XGB[data_knn>=0.2*iter]
    data_XGB.dropna(axis=1, inplace=True)
    data_LDA = data_LDA[data_knn>=0.2*iter]
    data_LDA.dropna(axis=1, inplace=True)
    data_nn = data_nn[data_knn>=0.2*iter]
    data_nn.dropna(axis=1, inplace=True)

    print(df)
    occurences = {str(ind):0 for ind in df.index}
    for ind in df.index:
        ind = str(ind)
        if ind in data_knn:
            occurences[ind] += 1
        if ind in data_LR:
            occurences[ind] += 1
        if ind in data_svc:
            occurences[ind] += 1
        if ind in data_XGB:
            occurences[ind] += 1
        if ind in data_LDA:
            occurences[ind] += 1
        if ind in data_nn:
            occurences[ind] += 1

    print(reverse_sort(occurences))

#class errors
if class_err == 1:
    print(data_mat_knn)
    print(data_mat_LR)
    print(data_mat_svc)
    print(data_mat_XGB)
    print(data_mat_LDA)
    print(data_mat_nn)


#Print
if print == 1:
    print(data_knn)
    print(data_LDA)
    print(data_LR)
    print(data_XGB)
    print(data_svc)
    print(data_nn)

#calulate accuracy
if accuracy == 1:
    print(acc)

#Show pictures
if pictures == 1:
    plot_ind(data_knn)
    plot_ind(data_LR)
    plot_ind(data_svc)
    plot_ind(data_XGB)
    plot_ind(data_LDA)
    plot_ind(data_nn)
    

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




