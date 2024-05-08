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


#plt.imshow(df.iloc[77].to_numpy().reshape(64,64).transpose(), cmap=colormaps['cool'])
#plt.show()

def pre_process(data, labels, train_size):
    #Split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=1-train_size)

    #scaler = StandardScaler()
    #X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    #X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return x_train, x_test, y_train, y_test


failures = dict()
mat =np.zeros((2,2))

iter = 1000
for j in tqdm(range(iter)):
    x_train, x_test, y_train, y_test = pre_process(df, labels_df, 0.7)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    #test = cross_val_predict(knn, x_train, y_train, cv =5, n_jobs=-1)

    y_pred = knn.predict(x_test)
    acc = sum(y_pred==y_test)/len(y_pred)
    #print(acc)

    mat += confusion_matrix(y_test,y_pred)

    fails = y_pred!=y_test
    for i in fails*x_test.index:
        if i != 0:
            if (str(i) in failures):
                failures[str(i)] += 1
            else:
                failures[str(i)] = 1
    #failure = [i for i in fails*x_test.index if i != 0 ]

#print(failures.items())

print(dict(reversed(sorted(failures.items(), key=lambda item: item[1]))))
#print(failures[max(failures, key=failures.get)])
print(mat/iter)


