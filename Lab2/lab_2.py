import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm




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
    max_trees = 50
    max_depth = 100
    RF_mean_scores = np.zeros(max_depth)
    RF_std_scores = np.zeros(max_depth)

    for depth in tqdm(range(max_depth)):
        RF = RandomForestClassifier(n_estimators = max_trees, max_depth=depth+1)
        
        RF_score = cross_val_score(RF, X_train, y_train, cv = 5)

        RF_mean_score = RF_score.mean()
        RF_std = RF_score.std()
        RF_mean_scores[depth] = RF_mean_score
        RF_std_scores[depth] = RF_std


    RF_optimal_depth = np.where(RF_mean_scores==RF_mean_scores.max())[0][0]+1
    RF_optimal_std = RF_std_scores[RF_optimal_depth-1]
    cross_val_err = 1 - max(RF_mean_scores)

    RF.fit(X_train, y_train)
    train_pred = RF.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = RF.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    er_clas =dict()
    for clas in classes:
        #print(y_test[classes[clas]])
        er_clas[clas] = 1-accuracy_score(y_test[classes[clas]] ,RF.predict(X_test.iloc[classes[clas]]))


    print("RF optimal depth:", RF_optimal_depth)
    print("Standard deviation of cross val error: ", RF_optimal_std)
    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)
    print("Class test error: ", er_clas)
    return [train_error, cross_val_err, RF_optimal_std, test_error, RF_optimal_depth, er_clas, list(data.columns[RF.feature_importances_>0].values), list(RF.feature_importances_[RF.feature_importances_>0])]

def gradient_boosting(X_train, X_test, y_train, y_test, classes, data):
    tree_sizes = [5]
    max_depth = 2
    GB_mean_scores = np.zeros(len(tree_sizes))
    GB_std_scores = np.zeros(len(tree_sizes))

    for i in tqdm(range(len(tree_sizes))):
        GB = GradientBoostingClassifier(n_estimators = tree_sizes[i], max_depth = max_depth)
        
        GB_score = cross_val_score(GB, X_train, y_train, cv = 2)

        GB_mean_score = GB_score.mean()
        GB_std = GB_score.std()
        GB_mean_scores[i] = GB_mean_score
        GB_std_scores[i] = GB_std


    GB_optimal_n_trees = tree_sizes[np.where(GB_mean_scores==GB_mean_scores.max())[0][0]]
    GB_optimal_std = GB_std_scores[np.where(GB_mean_scores==GB_mean_scores.max())[0][0]]
    cross_val_err = 1 - max(GB_mean_scores)

    GB.fit(X_train, y_train)
    train_pred = GB.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = GB.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    er_clas =dict()
    for clas in classes:
        er_clas[clas] = 1-accuracy_score(y_test[classes[clas]] ,GB.predict(X_test.iloc[classes[clas]]))

    print("GB optimal number trees:", GB_optimal_n_trees)
    print("Standard deviation of cross val error: ", GB_optimal_std)
    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)
    print("Class test error: ", er_clas)

    return [train_error, cross_val_err, GB_optimal_std, test_error, GB_optimal_n_trees, er_clas, list(data.columns[GB.feature_importances_>0].values), list(GB.feature_importances_[GB.feature_importances_>0])]



## Cancer dataset
df = pd.read_csv('./data/TCGAdata.txt', sep=" " ,header=0,index_col= 0)
labels_df = pd.read_csv('./data/TCGAlabels', sep=" " ,header=0, index_col= 0)

d= dict()
X_train, X_test, y_train, y_test, classes = pre_process(df, labels_df, 0.8)
'''

#Bagging
#d["Noise_0.0"] = random_forest(X_train, X_test, y_train, y_test, classes)
for error in [0,0.1,0.5,1,3]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = random_forest(X_train_noise, X_test_noise, y_train, y_test, classes, df)

 
df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross','std', 'Test', 'Depth' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data.csv', sep=" ")
'''

#Gradient boosting

for error in [0,0.1,0.3,0.5,0.8,1]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = gradient_boosting(X_train_noise, X_test_noise, y_train, y_test, classes, df)


df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross','std', 'Test', 'Trees' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data_gb.csv', sep=" ")


## Cats and dogs data set
df_images = pd.read_csv('./data/CATSnDOGS.csv', sep="," ,header=0,index_col= 0)
labels_df_images = pd.read_csv('./data/Labels.csv')


d_1= dict()
X_train, X_test, y_train, y_test, classes = pre_process(df_images, labels_df_images, 0.8)


#Bagging
#d["Noise_0.0"] = random_forest(X_train, X_test, y_train, y_test, classes)
for error in [0,0.1,0.5,1,3]:
    X_train_noise, X_test_noise = noise(X_train, X_test, noise = error)
    d_1[f"Noise_{error:.1f}"] = random_forest(X_train_noise, X_test_noise, y_train, y_test, classes, df_images)

 
df_1 = pd.DataFrame(data =d_1, index = ['Train', 'Cross','std', 'Test', 'Depth' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data_cat.csv', sep=" ")



#Gradient boosting

for error in [0,0.1,0.3,0.5,0.8,1]:
    X_train_noise, X_test_noise= noise(X_train, X_test, noise = error)
    d[f"Noise_{error:.1f}"] = gradient_boosting(X_train_noise, X_test_noise, y_train, y_test, classes, df_images)


df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross','std', 'Test', 'Trees' , 'Class_errors', 'Important_labels', 'Importance_value'])
df_1.to_csv('./data_gb_cat.csv', sep=" ")
