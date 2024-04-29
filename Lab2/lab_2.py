import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier


def pre_process(data, labels, train_size):
    #Split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(data, labels.values.ravel(), test_size=1-train_size)

    #Standardize the columns
    #Scale after split to avoid data leakage
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    classes = dict()
    for i in range(len(y_test)):
        if y_test[i] not in classes:
            classes[y_test[i]] = [i]
        else:
            classes[y_test[i]].append(i)
    
    #print(X_test.iloc[classes['U']])
    
    return X_train, X_test, y_train, y_test, classes


def noise(X_train, X_test, noise):
    X_train_noise = X_train.copy(deep=True)
    X_test_noise = X_test.copy(deep=True)
    X_train_noise += np.random.normal(loc = 0,scale = noise,size = [X_train.shape[0], X_train.shape[1]])
    X_test_noise += np.random.normal(loc = 0,scale =noise, size = [X_test.shape[0], X_test.shape[1]])
    return X_train_noise, X_test_noise


def random_forest(X_train, X_test, y_train, y_test, classes, data):
    nr_trees= [50, 75, 100, 125, 150]
    depth = [10, 25, 50, 75, 90]
    RF_mean_scores = np.zeros((len(nr_trees), len(depth)))

    for i in tqdm(range(len(nr_trees))):
        for j in tqdm(range(len(depth))):
            RF = RandomForestClassifier(n_estimators = nr_trees[i], max_depth=depth[j])
            
            RF_score = cross_val_score(RF, X_train, y_train, cv = 5)

            RF_mean_score = RF_score.mean()
            RF_mean_scores[depth] = RF_mean_score


    max_index = np.unravel_index(RF_mean_scores.argmax(), RF_mean_scores.shape)
    print(max_index)
    RF_optimal_n_trees = nr_trees[max_index[0]]
    RF_optimal_depth = depth[max_index[1]]

    cross_val_err = 1 - RF_mean_scores[max_index]

    RF.fit(X_train, y_train)
    train_pred = RF.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = RF.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    er_clas = dict()
    for clas in classes:
        er_clas[clas] = 1-accuracy_score(y_test[classes[clas]] ,RF.predict(X_test.iloc[classes[clas]]))

    print("RF optimal number of trees:", RF_optimal_n_trees)
    print("RF optimal depth:", RF_optimal_depth)
    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)
    print("Class test error: ", er_clas)
    return [train_error, cross_val_err, test_error, RF_optimal_n_trees, RF_optimal_depth, er_clas, list(data.columns[RF.feature_importances_>0].values), list(RF.feature_importances_[RF.feature_importances_>0])]


def gradient_boosting(X_train, X_test, y_train, y_test, classes, data):
    nr_tree = [50, 75, 100, 125, 150]
    learn_rate = [0.1, 0.2, 0.3]
    max_depth = 3
    GB_mean_scores = np.zeros((len(nr_tree), len(learn_rate)))

    for i in tqdm(range(len(nr_tree))):
        for j in tqdm(range(len(learn_rate))):
            GB = XGBClassifier(n_estimators = nr_tree[i], learning_rate = learn_rate[j], max_depth = max_depth)
            
            GB_score = cross_val_score(GB, X_train, y_train, cv = 5)

            GB_mean_score = GB_score.mean()
            GB_mean_scores[i][j] = GB_mean_score
            print(GB_mean_scores)

    max_index = np.unravel_index(GB_mean_scores.argmax(), GB_mean_scores.shape)
    print(max_index)
    GB_optimal_n_trees = nr_tree[max_index[0]]
    GB_optimal_learn_rate = learn_rate[max_index[1]]
    
    cross_val_err = 1 - GB_mean_scores[max_index]

    GB.fit(X_train, y_train)
    train_pred = GB.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = GB.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    er_clas = dict()
    for clas in classes:
        er_clas[clas] = 1-accuracy_score(y_test[classes[clas]] ,GB.predict(X_test.iloc[classes[clas]]))

    print("GB optimal number trees:", GB_optimal_n_trees)
    print("GB optimal learning rate:", GB_optimal_learn_rate)
    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)
    print("Class test error: ", er_clas)

    return [train_error, cross_val_err, test_error, GB_optimal_n_trees, GB_optimal_learn_rate, er_clas, list(data.columns[GB.feature_importances_>0].values), list(GB.feature_importances_[GB.feature_importances_>0])]


## Cancer dataset
df = pd.read_csv('./data/TCGAdata.txt', sep=" " ,header=0,index_col= 0)
labels_df = pd.read_csv('./data/TCGAlabels', sep=" " ,header=0, index_col= 0)


X_train, X_test, y_train, y_test, classes = pre_process(df, labels_df, 0.8)


#Bagging
d = dict()
for error in [0, 0.1, 0.5, 1, 3]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = random_forest(X_train_noise, X_test_noise, y_train, y_test, classes, df)

 
df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross','std', 'Test', 'Trees', 'Depth' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data.csv', sep=" ")


#Gradient boosting
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

d = dict()
for error in [0, 0.1, 0.5, 1, 3]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = gradient_boosting(X_train_noise, X_test_noise, y_train, y_test, classes, df)


df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross', 'Test', 'Trees', 'Learn_rate' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data_gb.csv', sep=" ")


## Cats and dogs data set
df_images = pd.read_csv('./data/CATSnDOGS.csv', sep="," ,header=0,index_col= 0)
labels_df_images = pd.read_csv('./data/Labels.csv')

X_train, X_test, y_train, y_test, classes = pre_process(df_images, labels_df_images, 0.8)


#Bagging
d = dict()
#d["Noise_0.0"] = random_forest(X_train, X_test, y_train, y_test, classes)
for error in [0, 0.1, 0.5, 1, 3]:
    X_train_noise, X_test_noise = noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = random_forest(X_train_noise, X_test_noise, y_train, y_test, classes, df_images)

 
df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross','Test','Trees',  'Depth' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data_cat.csv', sep=" ")



#Gradient boosting
d = dict()
for error in [0, 0.1, 0.5, 1, 3]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = gradient_boosting(X_train_noise, X_test_noise, y_train, y_test, classes, df_images)


df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross', 'Test', 'Trees', 'Learn_rate' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data_gb_cat.csv', sep=" ")
