import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier




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

    acc = accuracy(y_pred, y_test)
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
LDA_failures = dict()

LR_mat = np.zeros((2,2))
knn_mat = np.zeros((2,2))
XGB_mat = np.zeros((2,2))
svc_mat = np.zeros((2,2))
LDA_mat = np.zeros((2,2))

for i in range(10):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)
    '''
    import torch
    import torch.nn as nn
    import torch.optim as optim
    ##Neural network

    model = nn.Sequential(
    nn.Linear(4096, 5000),
    nn.ReLU(),
    nn.Linear(5000,2000),
    nn.ReLU(),
    nn.Linear(2000, 80),
    nn.ReLU(),
    nn.Linear(80, 2),
    nn.Softmax(dim=1))



    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)
    x_nn = torch.tensor(x_train.to_numpy(), dtype=torch.float32, requires_grad = True)
    y_nn = torch.tensor(y_train, dtype=torch.float32, requires_grad = True)

    x_test_nn = torch.tensor(x_test.to_numpy(), dtype=torch.float32, requires_grad = True)
    y_test_nn = torch.tensor(y_test, dtype=torch.float32, requires_grad = True)


    for epoch in range(10):
        for i in range(2):
            x_1 = x_nn[i*69:(i+1)*69]
            y_1 = y_nn[i*69:(i+1)*69]
            y_pred = model(x_1)
            y = []
            for sub_list in y_pred:
                y.append(sub_list.argmax(dim=0).item())

            #optimizer.zero_grad()
            loss = loss_fn(torch.tensor(y, dtype=torch.float32), y_1)

            loss.backward()
            optimizer.step()    
        print(f'Finished epoch {epoch}, latest loss {loss}')


    with torch.no_grad():
        y_pred = model(x_test_nn)
    y = []
    for sub_list in y_pred:
        y.append(sub_list.argmax(dim=0).item())
    accuracy = sum(torch.tensor(y) == y_test_nn)/len(y_test)
    print(f"Accuracy {accuracy}")
'''


iter = 100
for j in tqdm(range(iter)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)
    ## LDA
    LDA = LinearDiscriminantAnalysis()
    classifier(LDA, LR_mat, LDA_failures)

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
LDA_failures = reverse_sort(LDA_failures)

pd.DataFrame(data = knn_failures, index=[0]).to_csv('./data_knn', sep = " ")
pd.DataFrame(data = LR_failures, index=[0]).to_csv('./data_LR', sep = " ")
pd.DataFrame(data = XGB_failures, index=[0]).to_csv('./data_XGB', sep = " ")
pd.DataFrame(data = svc_failures, index=[0]).to_csv('./data_svc', sep = " " )
pd.DataFrame(data = LDA_failures, index=[0]).to_csv('./data_LDA', sep = " ")
 
pd.DataFrame(data = LR_mat/iter,index=[0,1]).to_csv('./data_mat_LR', sep = " ")
pd.DataFrame(data = knn_mat/iter,index=[0,1]).to_csv('./data_mat_knn', sep = " ")
pd.DataFrame(data = XGB_mat/iter,index=[0,1]).to_csv('./data_mat_XGB', sep = " ")
pd.DataFrame(data = svc_mat/iter,index=[0,1]).to_csv('./data_mat_svc', sep = " ")
pd.DataFrame(data = LDA_mat/iter,index=[0,1]).to_csv('./data_mat_LDA', sep = " ")




