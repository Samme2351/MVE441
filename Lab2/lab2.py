import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm



def test_split(split, df, labels_df):
    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-split, random_state=42)
    
    return X_train, X_test, y_train, y_test

def rf(X_train, X_test, y_train, y_test):
    max_trees = 100
    mean_errors = np.zeros(max_trees)
    std_devs = np.zeros(max_trees)
    top10_feat = np.zeros((max_trees,10), dtype=tuple)
    class_error_test = np.zeros((max_trees, 6))

    for i in tqdm(range(max_trees)):
        forest = RandomForestClassifier(n_estimators=i+1)
        forest.fit(X_train, y_train)

        scores = cross_val_score(forest, X_train, y_train)
        mean_errors[i] = np.mean(scores)
        std_devs[i] = np.std(scores)

        en_importance = list(enumerate(forest.feature_importances_))
        en_importance.sort(key = lambda a:a[1], reverse= True)
        top10_feat[i] = en_importance[:10]

        f_pred = forest.predict(X_test)
        confusion = confusion_matrix(y_test, f_pred)

        for j in range(6):
            class_error_test[i,j] = confusion[j][j]/sum(confusion[:][j])


    return mean_errors, std_devs, top10_feat, class_error_test

def bg(X_train, X_test, y_train, y_test):

    return 0

def noisifyer(mean, std, df):
    dataframe_size = df.shape
    noise = np.random.default_rng().normal(mean, std, size = dataframe_size)

    return df.add(noise)

def plot_feature(a,b, df, labels_df, ax):
    
    cmap = colormaps['cool']
    cancerlist = labels_df.values.ravel()
    classes = set(cancerlist)

    color_dict = {}
    for i,cla in enumerate(classes):
        color_dict[cla] = cmap(i/len(classes))

    cl = np.zeros(len(labels_df),dtype=tuple)

    for i in range(len(labels_df)):
        cl[i] = color_dict[cancerlist[i]]

    ax.scatter(df['V'+str(a)],df['V'+str(b)], c = cl)

    return 0

def main():
    split = 0.7
    df = pd.read_csv('data/TCGAdata.txt', sep = " ", header = 0)
    labels_df = pd.read_csv('data/TCGALabels', sep = " ", header = 0)
    fps = (4,4)


    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-split, random_state=42)

    figure1, axis = plt.subplots(fps[0], fps[1]) 
    for i in range(fps[0]):
        for j in range(fps[1]):
            plot_feature(i+1,j+1,df,labels_df, axis[i,j])

    #plt.show()


    mean_errors, std_devs, top10_feat, class_error_test = rf(X_train, X_test, y_train, y_test)
    
    print(class_error_test.shape)
    plotrange = range(1,101)

    print('Standard deviations are: ',std_devs)
    print('Top 10 features are: ', top10_feat)

    figure2 = plt.figure()
    plt.plot(plotrange, mean_errors)
  

    figure3 = plt.figure()

    for c in range(class_error_test.shape[1]):
        plt.plot(plotrange, class_error_test[:,c])

    plt.show()
    '''

    for s in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]:
        noisifyer(0, s, X_train)
        noisifyer(0, s, X_test)
        forest = RandomForestClassifier()
        forest.fit(X_train, y_train)

        train_pred = forest.predict(X_train)
        train_err = 1- accuracy_score(y_train, train_pred)

        test_pred = forest.predict(X_test)
        test_err = 1- accuracy_score(y_test, test_pred)
   
    print("train error: ", train_err)
    print("test error: ", test_err)

    print(forest.feature_importances_)
    #print(sum(forest.feature_importances_))

    en = list(enumerate(forest.feature_importances_))

    en.sort(key = lambda a:a[1], reverse= True)
    
    #print(en)
    print(type(en[0:10]))
    '''
    return 0

main()