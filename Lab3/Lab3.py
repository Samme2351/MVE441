import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import load_digits

df = pd.read_csv('../Data/CATSnDOGS.csv', sep="," ,header = 0)
labels_df = pd.read_csv('../Data/Labels.csv', header = 0)


#plt.imshow(df.iloc[77].to_numpy().reshape(64,64).transpose(), cmap=colormaps['cool'])
#plt.show()

def pre_process(data, labels, train_size):
    #Split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=1-train_size)

    '''
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    '''

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

'''
def reverse_sort(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

LR_failures = dict()
knn_failures = dict()
XGB_failures = dict()
svc_failures = dict()

LR_mat = np.zeros((2,2))
knn_mat = np.zeros((2,2))
XGB_mat = np.zeros((2,2))
svc_mat = np.zeros((2,2))

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
    

    loss_fn = nn.LogSoftmax()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    x_nn = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    y_nn =torch.tensor(y_train, dtype=torch.float32)
    print(x_nn)


    for epoch in range(100):
        y_pred = model(x_nn)
        print(y_pred)
        y = []
        for sub_list in y_pred:
            y.append(sub_list.argmax(dim=0).item())
        print(y)
        loss = loss_fn(y, y_nn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

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
'''


## 1 b
for n in tqdm(range(10)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)

    LR = LogisticRegression(penalty='l1', solver='liblinear', max_iter=300)
    LR.fit(x_train, y_train)
    LR_coef = pd.DataFrame(data = LR.coef_[0,:])
    LR_coef.to_csv('./data_1b_lr_' + str(n), sep = " ")
    #abs_coef = abs(LR_coef)
    #Z = zip(abs_coef, [x for x in range(4096)])
    #LR_imp_features = [x for _, x in sorted(zip(abs_coef, [x for x in range(4096)]), key=lambda pair: pair[0], reverse=True)][:100]
    #print(LR_imp_features)
    #print(abs_coef[LR_imp_features])

    feature_selector = SelectKBest(f_classif, k=100)
    x_train_selected = feature_selector.fit_transform(x_train, y_train)
    #print(x_train_selected.shape)
    #print(df.columns[feature_selector.get_support()])
    #KB_features = df.columns
    KB_scores = pd.DataFrame(feature_selector.scores_)
    #KB_results = pd.DataFrame([KB_features, KB_scores])
    #KB_results.to_csv('./data_1b_kb_' + str(n), sep = " ")
    KB_scores.to_csv('./data_1b_kb_' + str(n), sep = " ")

    NC = NearestCentroid(shrink_threshold=0.5)
    NC.fit(x_train, y_train)
    NC_overall_centroid = (NC.centroids_[0,:] + NC.centroids_[1,:])/2
    #print(abs(NC.centroids_[0]-NC_overall_centroid).max())
    NC_centroids = pd.DataFrame(NC.centroids_)
    NC_centroids.to_csv('./data_1b_nc_' + str(n), sep = " ")


## 2
tcga_df = pd.read_csv('../Data/TCGAdata.txt', sep=" " ,header = 0)
tcga_labels = pd.read_csv('../Data/TCGAlabels', sep = " ", header = 0)

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