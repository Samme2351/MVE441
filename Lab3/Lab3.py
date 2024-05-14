import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

def classifier(model, mat, failures, acc, name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc[name] += accuracy_score(y_true=y_test, y_pred=y_pred)/iter
    mat += confusion_matrix(y_true=y_test, y_pred=y_pred)/iter

    fails = y_pred!=y_test
    for i in fails*x_test.index:
        if i != 0:
            failures[i] = failures.get(i,0) + 1

    return model, mat, failures, acc


def reverse_sort(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 200)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


LR_failures = dict()
knn_failures = dict()
XGB_failures = dict()
svc_failures = dict()
LDA_failures = dict()
nn_failures = dict()

LR_mat = np.zeros((2,2))
knn_mat = np.zeros((2,2))
XGB_mat = np.zeros((2,2))
svc_mat = np.zeros((2,2))
LDA_mat = np.zeros((2,2))
nn_mat = np.zeros((2,2))

acc = {"LR": 0, "knn": 0, "XGB": 0, "svc": 0, "LDA": 0, "nn": 0}


iter = 10
for j in tqdm(range(iter)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)

    ## NN
    x_train_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
    x_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  

    model = BinaryClassifier()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00002)

    epochs = 500
    for epoch in range(epochs):  
        optimizer.zero_grad()
        
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        #print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        outputs = model(x_test_tensor)
        y_pred = (outputs > 0.5)#.float()
        #accuracy = (y_pred == y_test_tensor).float().mean()
        #print(f'Accuracy: {accuracy.item():.4f}')
        acc["nn"] += accuracy_score(y_true=y_test_tensor, y_pred=y_pred)/iter
        nn_mat += confusion_matrix(y_true=y_test_tensor, y_pred=y_pred)/iter

    fails = y_pred!=y_test
    for i in fails*x_test.index:
        if i != 0:
            nn_failures[i] = nn_failures.get(i,0) + 1
    ## LDA
    LDA = LinearDiscriminantAnalysis()
    classifier(LDA, LDA_mat, LDA_failures, acc, "LDA")

    ## SVC
    svc = SVC()
    classifier(svc, svc_mat, svc_failures, acc, "svc")

    ## XGBoost
    XGB = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
    classifier(XGB, XGB_mat, XGB_failures, acc, "XGB")


    ## Logistic regression
    LR = LogisticRegression(penalty='l1', solver='liblinear', max_iter=300)
    classifier(LR, LR_mat, LR_failures, acc, "LR")

    ## KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    classifier(knn, knn_mat, knn_failures, acc, "knn")

pd.DataFrame(data =acc, index=[0]).to_csv('./data_acc', sep= " ")

pd.DataFrame(data = reverse_sort(knn_failures), index=[0]).to_csv('./data_knn', sep = " ")
pd.DataFrame(data = reverse_sort(LR_failures), index=[0]).to_csv('./data_LR', sep = " ")
pd.DataFrame(data = reverse_sort(XGB_failures), index=[0]).to_csv('./data_XGB', sep = " ")
pd.DataFrame(data = reverse_sort(svc_failures), index=[0]).to_csv('./data_svc', sep = " " )
pd.DataFrame(data = reverse_sort(LDA_failures), index=[0]).to_csv('./data_LDA', sep = " ")
pd.DataFrame(data = reverse_sort(nn_failures), index=[0]).to_csv('./data_nn', sep = " ")
 
pd.DataFrame(data = LR_mat/iter,index=[0,1]).to_csv('./data_mat_LR', sep = " ")
pd.DataFrame(data = knn_mat/iter,index=[0,1]).to_csv('./data_mat_knn', sep = " ")
pd.DataFrame(data = XGB_mat/iter,index=[0,1]).to_csv('./data_mat_XGB', sep = " ")
pd.DataFrame(data = svc_mat/iter,index=[0,1]).to_csv('./data_mat_svc', sep = " ")
pd.DataFrame(data = LDA_mat/iter,index=[0,1]).to_csv('./data_mat_LDA', sep = " ")
pd.DataFrame(data = nn_mat/iter,index=[0,1]).to_csv('./data_mat_nn', sep = " ")






