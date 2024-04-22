import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm


df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)
max_num_components = 40
num_components_range = range(1, max_num_components)
train_size = 0.7

def test_split(split, df, labels_df):
    train_size = split
    # Sepearate into test data and working data
    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-train_size, random_state=42)
    
    #Standardize the rows (transposing as fit_transform standardizes along columns)
    #Scale after split to avoid data leakage
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train, X_test, y_train, y_test



def KNN_PCA_2(X_train, X_test, y_train, y_test):

    KNN_mean_scores = np.zeros(max_num_components)
    KNN_train_errors = np.zeros(max_num_components)
    KNN_test_errors = np.zeros(max_num_components)

    # Loop over different numbers of components
    for n_components in tqdm(num_components_range):
        #PCA

        KNN_pipeline = make_pipeline(PCA(n_components=n_components), KNeighborsClassifier(n_neighbors=5))
        #KNN_pipeline = make_pipeline(PCA(n_components=n_components), SVC())
        #KNN_pipeline = make_pipeline(PCA(n_components=n_components), LogisticRegression(random_state=16, max_iter=10000))

        KNN_scores = cross_val_score(KNN_pipeline, X_train, y_train, cv=5)
        KNN_mean_score = KNN_scores.mean()

        #KNN_mean_scores.append(KNN_mean_score)
        KNN_mean_scores[n_components] = KNN_mean_score
        KNN_pipeline.fit(X_train, y_train)
        
        train_pred = KNN_pipeline.predict(X_train)
        KNN_train_errors[n_components] = 1 - accuracy_score(y_train, train_pred)
        
        test_pred = KNN_pipeline.predict(X_test)
        KNN_test_errors[n_components] = 1 - accuracy_score(y_test, test_pred)

    np.savetxt('KNN_err.csv', (KNN_mean_scores, KNN_train_errors, KNN_test_errors), delimiter=',')

# Split test
def KNN_PCA_3(df, labels_df):
    split_range = np.arange(0.1, 1, 0.05)
    KNN_mean_scores = np.zeros(len(split_range))
    KNN_train_errors = np.zeros(len(split_range))
    KNN_test_errors = np.zeros(len(split_range))
    iter = 0

    # Loop over different numbers of components
    for percentage in tqdm(split_range):
        X_train, X_test, y_train, y_test = test_split(percentage, df ,labels_df)

        #PCA
        #KNN_pipeline = make_pipeline(PCA(n_components=15), KNeighborsClassifier(n_neighbors=5))
        KNN_pipeline = make_pipeline(PCA(n_components=15), SVC())
        #KNN_pipeline = make_pipeline(PCA(n_components=25), LogisticRegression(random_state=16, max_iter=10000))

        KNN_scores = cross_val_score(KNN_pipeline, X_train, y_train, cv=5)
        KNN_mean_score = 1 - KNN_scores.mean()
        KNN_mean_scores[iter] = KNN_mean_score

        KNN_pipeline.fit(X_train, y_train)
        
        train_pred = KNN_pipeline.predict(X_train)
        KNN_train_errors[iter] = 1 - accuracy_score(y_train, train_pred)
        
        test_pred = KNN_pipeline.predict(X_test)
        KNN_test_errors[iter] = 1 - accuracy_score(y_test, test_pred)
        iter +=1

    np.savetxt('KNN_err_splits.csv', (KNN_mean_scores, KNN_train_errors, KNN_test_errors), delimiter=',')


    ##KNN features

def Features_2(X_train, X_test, y_train, y_test):
    max_num_features = 100

    num_features = range(1, max_num_features)
    KNN_mean_scores = np.zeros(max_num_features)
    KNN_train_errors = np.zeros(max_num_features)
    KNN_test_errors = np.zeros(max_num_features)

    # Loop over different numbers of features
    for k in tqdm(num_features):

        feature_selector = SelectKBest(f_classif, k=k)

        X_train_selected = feature_selector.fit_transform(X_train, y_train)

        #model = KNeighborsClassifier(n_neighbors=5)
        #model = SVC()
        model = LogisticRegression(random_state=16, max_iter=10000)

        KNN_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
        KNN_mean_score = KNN_scores.mean()

        KNN_mean_scores[k] = 1-KNN_mean_score

        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        model.fit(X_train_selected, y_train)

        train_pred = model.predict(X_train_selected)
        KNN_train_errors[k] = 1 - accuracy_score(y_train, train_pred)
            
        selected_feature_indices = feature_selector.get_support(indices=True)

        test_pred = model.predict(X_test[X_test.columns[selected_feature_indices]])
        KNN_test_errors[k] = 1 - accuracy_score(y_test, test_pred)


    np.savetxt('LR_err_feat.csv', (KNN_mean_scores, KNN_train_errors, KNN_test_errors), delimiter=',')


def main():
    df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
    labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)

    '''
    KNN_PCA_3(df, labels_df)

    plot_data = np.genfromtxt('KNN_err_splits.csv', delimiter=',')
 
    
    plot_range = np.arange(0.1, 1, 0.05)

    plt.plot(plot_range, plot_data[0])
    plt.plot(plot_range, plot_data[1])
    plt.plot(plot_range, plot_data[2])
    plt.legend(['Cross val error', 'Training error', 'Test error'])
    plt.xlabel('Training split')
    plt.ylabel('error')



    plt.show() 
    '''

    '''
    X_train, X_test, y_train, y_test = test_split(0.7, df ,labels_df)

    KNN_PCA_2(X_train, X_test, y_train, y_test)


    plot_data = np.genfromtxt('LR_err.csv', delimiter=',')
 
    one = np.ones(max_num_components)

    plot_range = range(1, max_num_components)
    plt.plot(plot_range, 1-plot_data[0][1:])
    plt.plot(plot_range, plot_data[1][1:])
    plt.plot(plot_range, plot_data[2][1:])

    plt.legend(['Mean error', 'Training error', 'Test error'])
    plt.xlabel('n PCA')
    plt.ylabel('error')

    plt.show() 
    '''

    '''
    KNN_features(X_train, X_test, y_train, y_test)
    SVC_PCA(X_train, X_test, y_train, y_test)
    SVC_features(X_train, X_test, y_train, y_test)
    LR_PCA(X_train, X_test, y_train, y_test)
    LR_features(X_train, X_test, y_train, y_test)
    '''

    X_train, X_test, y_train, y_test = test_split(0.7, df ,labels_df)

    Features_2(X_train, X_test, y_train, y_test)

    plot_data = np.genfromtxt('LR_err_feat.csv', delimiter=',')
 
    plot_range = range(1, 100)
    plt.plot(plot_range, plot_data[0][1:])
    plt.plot(plot_range, plot_data[1][1:])
    plt.plot(plot_range, plot_data[2][1:])

    plt.legend(['Mean error', 'Training error', 'Test error'])
    plt.xlabel('n features for LR')
    plt.ylabel('error')

    plt.show() 
    return 0

main()