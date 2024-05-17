import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
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

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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




## 1 b
for n in tqdm(range(100)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)

    LR = LogisticRegression(penalty='l1', solver='liblinear', max_iter=300)
    LR.fit(x_train, y_train)
    LR_coef = pd.DataFrame(data = LR.coef_[0,:])
    LR_coef.to_csv('./data_1b_lr_' + str(n), sep = " ")

    perm = permutation_importance(LR, x_test y_test, n_repeats = 30)

    feature_selector = SelectKBest(f_classif, k=100)
    x_train_selected = feature_selector.fit_transform(x_train, y_train)
    KB_scores = pd.DataFrame(feature_selector.scores_)
    KB_scores.to_csv('./data_1b_kb_' + str(n), sep = " ")

    NC = NearestCentroid(shrink_threshold=0.5)
    NC.fit(x_train, y_train)
    print(NC.centroids_)
    NC_overall_centroid = (NC.centroids_[0,:] + NC.centroids_[1,:])/2
    NC_centroids = pd.DataFrame(NC.centroids_)
    NC_centroids.to_csv('./data_1b_nc_' + str(n), sep = " ")




## 2
tcga_df = pd.read_csv('./Data/TCGAdata.txt', sep=" " ,header = 0)
tcga_labels = pd.read_csv('./Data/TCGAlabels', sep = " ", header = 0)

digits_df, digits_labels = load_digits(return_X_y=True, as_frame=True)
splits = [0.1, 0.4, 0.7]

for n in tqdm(range(10)):
    for split in splits:
        x_train, x_test, y_train, y_test = pre_process(df, labels_df, split)

        ## KNN
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        knn_score = f1_score(y_test, y_pred, average=None)
        knn_score = np.append(knn_score, accuracy_score(y_test, y_pred))

        ## LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)
        lda_score = f1_score(y_test, y_pred, average=None)
        lda_score = np.append(lda_score, accuracy_score(y_test, y_pred))

        ## QDA
        qda = QuadraticDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        y_pred = lda.predict(x_test)
        qda_score = f1_score(y_test, y_pred, average=None)
        qda_score = np.append(qda_score, accuracy_score(y_test, y_pred))

        ## LR
        lr = LogisticRegression(solver='liblinear', max_iter=300)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        lr_score = f1_score(y_test, y_pred, average=None)
        lr_score = np.append(lr_score, accuracy_score(y_test, y_pred))

        ## Lasso
        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=300)
        lasso.fit(x_train, y_train)
        y_pred = lasso.predict(x_test)
        lasso_score = f1_score(y_test, y_pred, average=None)
        lasso_score = np.append(lasso_score, accuracy_score(y_test, y_pred))

        ## SVC
        svc = SVC()
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        svc_score = f1_score(y_test, y_pred, average=None)
        svc_score = np.append(svc_score, accuracy_score(y_test, y_pred))

        ## RF
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        rf_score = f1_score(y_test, y_pred, average=None)
        rf_score = np.append(rf_score, accuracy_score(y_test, y_pred))

        ## XGB
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)
        y_pred = xgb.predict(x_test)
        xgb_score = f1_score(y_test, y_pred, average=None)
        xgb_score = np.append(xgb_score, accuracy_score(y_test, y_pred))

        scores = pd.DataFrame(data = [knn_score, lda_score, lr_score, lasso_score, svc_score, rf_score, xgb_score], index = ['knn', 'lda', 'lr', 'lasso', 'svc', 'rf', 'xgb'], columns = ['cat', 'dog', 'overall'])
        file_name = './data_2_' + str(split) + '_' + str(n)
        scores.to_csv(file_name, sep = ' ')






'''
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


iter = 1000
for j in tqdm(range(iter)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)

    ## NN
    x_train_tensor = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
    x_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  

    model = Classifier()

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
 
pd.DataFrame(data = LR_mat,index=[0,1]).to_csv('./data_mat_LR', sep = " ")
pd.DataFrame(data = knn_mat,index=[0,1]).to_csv('./data_mat_knn', sep = " ")
pd.DataFrame(data = XGB_mat,index=[0,1]).to_csv('./data_mat_XGB', sep = " ")
pd.DataFrame(data = svc_mat,index=[0,1]).to_csv('./data_mat_svc', sep = " ")
pd.DataFrame(data = LDA_mat,index=[0,1]).to_csv('./data_mat_LDA', sep = " ")
pd.DataFrame(data = nn_mat,index=[0,1]).to_csv('./data_mat_nn', sep = " ")
'''

