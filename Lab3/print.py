import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


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
accuracy = 1
class_err = 1
err_cut_off = 1

iter = 1000

if err_cut_off == 1:
    cut_off = 0.05
    print("During {iter} rounds with {cut_off*100:.0f}% cutoff")

    data_knn = data_knn[data_knn>=cut_off*iter]
    data_knn.dropna(axis=1, inplace=True)
    print(f"knn fails at {data_knn.shape[1]} of 198 pictures")
    data_LR = data_LR[data_knn>=cut_off*iter]
    data_LR.dropna(axis=1, inplace=True)
    print(f"LR fails at {data_LR.shape[1]} of 198 pictures")
    data_svc = data_svc[data_knn>=cut_off*iter]
    data_svc.dropna(axis=1, inplace=True)
    print(f"svc fails at {data_svc.shape[1]} of 198 pictures ")
    data_XGB = data_XGB[data_knn>=cut_off*iter]
    data_XGB.dropna(axis=1, inplace=True)
    print(f"XGB fails at {data_XGB.shape[1]} of 198 pictures ")
    data_LDA = data_LDA[data_knn>=cut_off*iter]
    data_LDA.dropna(axis=1, inplace=True)
    print(f"LDA fails at {data_LDA.shape[1]} of 198 pictures ")
    data_nn = data_nn[data_knn>=cut_off*iter]
    data_nn.dropna(axis=1, inplace=True)
    print(f"nn fails at {data_nn.shape[1]} of 198 pictures ")

    occurences = dict()
    for ind in df.index:
        ind = str(ind)
        if ind in data_knn:
            if ind in occurences:
                occurences[ind] += 1
            else:
                occurences[ind] = 1
        if ind in data_LR:
            if ind in occurences:
                occurences[ind] += 1
            else:
                occurences[ind] = 1
        if ind in data_svc:
            if ind in occurences:
                occurences[ind] += 1
            else:
                occurences[ind] = 1
        if ind in data_XGB:
            if ind in occurences:
                occurences[ind] += 1
            else:
                occurences[ind] = 1
        if ind in data_LDA:
            if ind in occurences:
                occurences[ind] += 1
            else:
                occurences[ind] = 1
        if ind in data_nn:
            if ind in occurences:
                occurences[ind] += 1
            else:
                occurences[ind] = 1

    print(f"Together all methods fail att {len(occurences)} out of the 198 pictures")
    print(reverse_sort(occurences))
    occur_6 = {i:occurences[i] for i in occurences if occurences[i]==6}
    print(reverse_sort(occur_6))
    print(f"All 6 methods fail att {len(occur_6)} out of 198 pictures")
    #plot_ind(pd.DataFrame(data=reverse_sort(occurences), index=[0]))

#class errors
if class_err == 1:
    print("KNN:")
    print(data_mat_knn)
    print("LR:")
    print(data_mat_LR)
    print("svc:")
    print(data_mat_svc)
    print("XGB:")
    print(data_mat_XGB)
    print("LDA:")
    print(data_mat_LDA)
    print("NN:")
    print(data_mat_nn)


#calulate accuracy
if accuracy == 1:
    print("Accuracy: ")
    print(acc)

#Show pictures
if pictures == 1:
    plot_ind(data_knn)
    plot_ind(data_LR)
    plot_ind(data_svc)
    plot_ind(data_XGB)
    plot_ind(data_LDA)
    plot_ind(data_nn)
    






