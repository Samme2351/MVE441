import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('./Data/CATSnDOGS.csv', sep="," ,header = 0)
labels_df = pd.read_csv('./Data/Labels.csv', header = 0)


#plt.imshow(df.iloc[77].to_numpy().reshape(64,64).transpose(), cmap=colormaps['cool'])
#plt.show()

def pre_process(data, labels, train_size):
    #Split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=1-train_size)

    return x_train, x_test, y_train, y_test

def classifier(model, mat, failures):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mat += confusion_matrix(y_true=y_test, y_pred=y_pred)

    fails = y_pred!=y_test
    for i in fails*x_test.index:
        if i != 0:
            failures[i] = failures.get(i,0) + 1

    return model, mat, failures

def reverse_sort(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

LR_failures = dict()
knn_failures = dict()
XGB_failures = dict()
svc_failures = dict()
nn_failures = dict()

LR_mat = np.zeros((2,2))
knn_mat = np.zeros((2,2))
XGB_mat = np.zeros((2,2))
svc_mat = np.zeros((2,2))
nn_mat = np.zeros((2,2))

iter = 10
for j in tqdm(range(iter)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)
    print(x_train.to_numpy().shape)

    ##Neural network

    model = nn.Sequential(
    nn.Linear(4096, 500),
    nn.ReLU(),
    nn.Linear(500,8),
    nn.ReLU(),
    nn.Linear(8, 2),
    nn.Softmax(dim=1))
    

    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    x_nn = torch.tensor(x_train.to_numpy(), dtype=torch.float32, requires_grad = True)
    y_nn =torch.tensor(y_train, dtype=torch.float32, requires_grad = True)

    '''
    for epoch in range(100):
        y_pred = model(x_nn)
        print(y_pred)
        y = []
        for sub_list in y_pred:
            y.append(sub_list.argmax(dim=0).item())
        print(torch.tensor(y))
        print(y_nn)
        loss = loss_fn(torch.tensor(y, dtype=torch.float32), y_nn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        print(f'Finished epoch {epoch}, latest loss {loss}')

    '''
    ## SVC
    svc = SVC()
    classifier(svc, svc_mat, svc_failures)

    ## XGBoost
    XGB = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
    classifier(XGB, XGB_mat, XGB_failures)


    ## Logistic regression
    LR = LogisticRegression(penalty='l1', solver='liblinear', max_iter=300)
    classifier(LR, LR_mat, LR_failures)

    ## KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    classifier(knn, knn_mat, knn_failures)


knn_failures = reverse_sort(knn_failures)
LR_failures = reverse_sort(LR_failures)
XGB_failures = reverse_sort(XGB_failures)
svc_failures = reverse_sort(svc_failures)

knn_fails = pd.DataFrame(data = knn_failures, index=[0])
LR_fails = pd.DataFrame(data = LR_failures, index=[0])
XGB_fails = pd.DataFrame(data = XGB_failures, index=[0])
svc_fails = pd.DataFrame(data = svc_failures, index=[0])
 
knn_fails.to_csv('./data_knn', sep = " ")
LR_fails.to_csv('./data_LR', sep = " ")
XGB_fails.to_csv('./data_XGB', sep = " ")
svc_fails.to_csv('./data_svc', sep = " " )


print(LR_mat/iter)
print(knn_mat/iter)
print(XGB_mat/iter)
print(svc_mat/iter)


