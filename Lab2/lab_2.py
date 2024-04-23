import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


df = pd.read_csv('./data/TCGAdata.txt', sep=" " ,header=0,index_col= 0)
labels_df = pd.read_csv('./data/TCGAlabels', sep=" " ,header=0, index_col= 0)


def pre_process(data, labels, train_size):
    #Split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=1-train_size)
    return X_train, X_test, y_train, y_test

def noise(X_train, X_test, noise):
    X_train_noise = X_train.copy(deep=True)
    X_test_noise = X_test.copy(deep=True)
    X_train_noise += np.random.normal(loc = 0,scale = noise,size = [X_train.shape[0], X_train.shape[1]])
    X_test_noise += np.random.normal(loc = 0,scale =noise, size = [X_test.shape[0], X_test.shape[1]])
    return X_train_noise, X_test_noise


def random_forest(X_train, X_test, y_train, y_test):
    max_trees = 1
    RF_mean_scores = np.zeros(max_trees)
    RF_std_scores = np.zeros(max_trees)

    for n_trees in tqdm(range(max_trees)):
        RF = RandomForestClassifier(n_estimators = n_trees+1)
        
        RF_score = cross_val_score(RF, X_train, y_train)

        RF_mean_score = RF_score.mean()
        RF_std = RF_score.std()
        RF_mean_scores[n_trees] = RF_mean_score
        RF_std_scores[n_trees] = RF_std


    RF_optimal_n_trees = np.where(RF_mean_scores==RF_mean_scores.max())[0][0]+1
    RF_optimal_std = RF_std_scores[RF_optimal_n_trees-1]
    cross_val_err = 1 - max(RF_mean_scores)

    RF.fit(X_train, y_train)
    train_pred = RF.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = RF.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    #print("RF optimal number of trees:", RF_optimal_n_trees)
    #print("Standard deviation of cross val error: ", RF_optimal_std)
    #print("Cross val err: ", cross_val_err)
    #print("Train err: ", train_error)
    #print("Train err: ", 1-RF.score(X_train, y_train))
    #print("Test err: ", test_error)
    #print("Test err: ",1-RF.score(X_test, y_test))
    #print(list(df.columns[RF.feature_importances_>0].values))
    return [train_error, cross_val_err, RF_optimal_std, test_error, list(df.columns[RF.feature_importances_>0].values), list(RF.feature_importances_[RF.feature_importances_>0])]

#Noise = 0
d= dict()
X_train, X_test, y_train, y_test = pre_process(df, labels_df, 0.8)


d["Noise_0"] = random_forest(X_train, X_test, y_train, y_test)

for error in [0.1,0.3,0.5,0.8,1]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = random_forest(X_train_noise, X_test_noise, y_train, y_test)



#print(d["test"])
#print(df.columns[imp>0])
#print(imp[imp>0])
#print(len(imp[imp>0]))



df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross','std', 'Test', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data.csv', sep=" ")