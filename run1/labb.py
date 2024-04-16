import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm

#features = [f"V{x}" for x in range(1,2000)]
df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)

#Set max number of components for PCA
max_num_components = 30

num_components_range = range(1, max_num_components)

#Pre-processes the data by splitting and normalizing 
def pre_process(data, labels, train_size):
    #Split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-train_size)
    
    #Standardize the rows (transposing as fit_transform standardizes along columns)
    #Scale after split to avoid data leakage
    scaler = StandardScaler()
    X_train = pd.DataFrame(np.transpose(scaler.fit_transform(X_train.transpose())), columns=X_train.columns)
    X_test = pd.DataFrame(np.transpose(scaler.fit_transform(X_test.transpose())), columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

##KNN PCA

def KNN_PCA(X_train, X_test, y_train, y_test):

    KNN_mean_scores = np.zeros(max_num_components)

    # Loop over different numbers of components
    for n_components in tqdm(num_components_range):

        #PCA

        KNN_pipeline = make_pipeline(PCA(n_components=n_components), KNeighborsClassifier(n_neighbors=5))

        KNN_scores = cross_val_score(KNN_pipeline, X_train, y_train, cv=5)
        KNN_mean_score = KNN_scores.mean()

        #KNN_mean_scores.append(KNN_mean_score)
        KNN_mean_scores[n_components] = KNN_mean_score

    KNN_optimal_n_components = np.where(KNN_mean_scores==KNN_mean_scores.max())[0][0]+1
    cross_val_err = 1 - max(KNN_mean_scores)

    print("KNN optimal number of PCA components:", KNN_optimal_n_components)

    opt_pipeline = make_pipeline(PCA(n_components=KNN_optimal_n_components), KNeighborsClassifier(n_neighbors=5))

    opt_pipeline.fit(X_train, y_train)
    train_pred = opt_pipeline.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = opt_pipeline.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)
    print("\n")

    return [train_error, cross_val_err, test_error, KNN_optimal_n_components]


##KNN features

def KNN_features(X_train, X_test, y_train, y_test):
    max_num_features = 100

    num_features = range(1, max_num_features)
    KNN_mean_scores = np.zeros(max_num_features)

    # Loop over different numbers of features
    for k in tqdm(num_features):

        feature_selector = SelectKBest(f_classif, k=k)

        X_train_selected = feature_selector.fit_transform(X_train, y_train)

        model = KNeighborsClassifier(n_neighbors=5)

        KNN_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
        KNN_mean_score = KNN_scores.mean()

        KNN_mean_scores[k] = KNN_mean_score

        #KNN_mean_scores.append(KNN_mean_score)

    KNN_optimal_k_features = np.where(KNN_mean_scores==KNN_mean_scores.max())[0][0]+1
    cross_val_err = 1 - max(KNN_mean_scores)

    print("KNN optimal number of features:", KNN_optimal_k_features)

    feature_selector = SelectKBest(f_classif, k=KNN_optimal_k_features)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    model.fit(X_train_selected, y_train)


    # Get the indices of the selected features
    selected_feature_indices = feature_selector.get_support(indices=True)

    # Get the names of the most predictive features
    selected_features = X_train.columns[selected_feature_indices]
    print("Most predictive features:", selected_features)


    train_pred = model.predict(X_train_selected)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test[X_test.columns[selected_feature_indices]])
    test_error = 1 - accuracy_score(y_test, test_pred)
    
    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)

    return [train_error, cross_val_err, test_error, KNN_optimal_k_features]

##SVC PCA
def SVC_PCA(X_train, X_test, y_train, y_test):
    SVC_mean_scores = np.zeros(max_num_components)

    #num_components_range = range(1, max_num_components)

    # Loop over different numbers of components
    for n_components in tqdm(num_components_range):

        #PCA
 
        SVC_pipeline = make_pipeline(PCA(n_components=n_components), SVC())

        SVC_scores = cross_val_score(SVC_pipeline, X_train, y_train, cv=5)
        SVC_mean_score = SVC_scores.mean()

        #SVC_mean_scores.append(SVC_mean_score)
        SVC_mean_scores[n_components] = SVC_mean_score

    SVC_optimal_n_components = np.where(SVC_mean_scores==SVC_mean_scores.max())[0][0]+1
    cross_val_err = 1 - max(SVC_mean_scores)

    print("SVC optimal number of PCA components:", SVC_optimal_n_components)

    opt_pipeline = make_pipeline(PCA(n_components=SVC_optimal_n_components), SVC())

    opt_pipeline.fit(X_train, y_train)
    train_pred = opt_pipeline.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = opt_pipeline.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)

    return [train_error, cross_val_err, test_error, SVC_optimal_n_components]##SVC features

def SVC_features(X_train, X_test, y_train, y_test):
    max_num_features = 100
    num_features = range(1, max_num_features)
    SVC_mean_scores = np.zeros(max_num_features)

    # Loop over different numbers of components
    for k in tqdm(num_features):

        feature_selector = SelectKBest(f_classif, k=k)

        X_train_selected = feature_selector.fit_transform(X_train, y_train)

        model = SVC()

        SVC_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
        SVC_mean_score = SVC_scores.mean()

        #SVC_mean_scores.append(SVC_mean_score)
        SVC_mean_scores[k] = SVC_mean_score


    SVC_optimal_k_features = np.where(SVC_mean_scores==SVC_mean_scores.max())[0][0]+1
    cross_val_err = 1 - max(SVC_mean_scores)

    print("SVC optimal number of features:", SVC_optimal_k_features)

    feature_selector = SelectKBest(f_classif, k=SVC_optimal_k_features)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    model.fit(X_train_selected, y_train)

    # Get the indices of the selected features
    selected_feature_indices = feature_selector.get_support(indices=True)

    # Get the names of the most predictive features
    selected_features = X_train.columns[selected_feature_indices]
    print("Most predictive features:", selected_features)


    train_pred = model.predict(X_train_selected)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test[X_test.columns[selected_feature_indices]])
    test_error = 1 - accuracy_score(y_test, test_pred)

    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)

    return [train_error, cross_val_err, test_error, SVC_optimal_k_features]



##Logistic regression PCA

def LR_PCA(X_train, X_test, y_train, y_test):
    LR_mean_scores = np.zeros(max_num_components)


    # Loop over different numbers of components
    for n_components in tqdm(num_components_range):

        #PCA

        LR_pipeline = make_pipeline(PCA(n_components=n_components), LogisticRegression(solver='lbfgs', max_iter=10000))

        LR_scores = cross_val_score(LR_pipeline, X_train, y_train, cv=5)
        LR_mean_score = LR_scores.mean()

        #LR_mean_scores.append(LR_mean_score)
        LR_mean_scores[n_components] = LR_mean_score

    LR_optimal_n_components = np.where(LR_mean_scores==LR_mean_scores.max())[0][0]+1
    cross_val_err = 1 - max(LR_mean_scores)

    print("KNN optimal number of PCA components:", LR_optimal_n_components)

    opt_pipeline = make_pipeline(PCA(n_components=LR_optimal_n_components), LogisticRegression(solver='lbfgs', max_iter=10000))

    opt_pipeline.fit(X_train, y_train)
    train_pred = opt_pipeline.predict(X_train)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = opt_pipeline.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)

    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)

    return [train_error, cross_val_err, test_error, LR_optimal_n_components]

##Logistic Regression features

def LR_features(X_train, X_test, y_train, y_test):
    max_num_features = 100
    num_features = range(1, max_num_features)
    LR_mean_scores = np.zeros(max_num_features)

    # Loop over different numbers of components
    for k in tqdm(num_features):

        feature_selector = SelectKBest(f_classif, k=k)

        X_train_selected = feature_selector.fit_transform(X_train, y_train)

        model = LogisticRegression(solver='lbfgs', max_iter=10000)

        LR_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
        LR_mean_score = LR_scores.mean()

        #LR_mean_scores.append(LR_mean_score)
        LR_mean_scores[k] = LR_mean_score

    LR_optimal_k_features = np.where(LR_mean_scores==LR_mean_scores.max())[0][0]+1
    cross_val_err = 1 - max(LR_mean_scores)

    print("LR optimal number of features:", LR_optimal_k_features)

    feature_selector = SelectKBest(f_classif, k=LR_optimal_k_features)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    model.fit(X_train_selected, y_train)

    # Get the indices of the selected features
    selected_feature_indices = feature_selector.get_support(indices=True)

    # Get the names of the most predictive features
    selected_features = X_train.columns[selected_feature_indices]
    print("Most predictive features:", selected_features)

    train_pred = model.predict(X_train_selected)
    train_error = 1 - accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test[X_test.columns[selected_feature_indices]])
    test_error = 1 - accuracy_score(y_test, test_pred)

    print("Cross val err: ", cross_val_err)
    print("Train err: ", train_error)
    print("Test err: ", test_error)

    return [train_error, cross_val_err, test_error, LR_optimal_k_features]

##Creating dictionary to save data of part 1
d = dict()
#a =0.7
#print(f"KNN_PCA_{(1-a)*100:.0f}_{a*100:.0f}")

##Run everything with 70/30 split
split = 0.7
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)


d[f"KNN_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train, y_test)
d[f"KNN_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train, y_test)
d[f"SVC_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train, y_test)
d[f"SVC_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train, y_test)
d[f"LR_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train, y_test)
d[f"LR_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train, y_test)


##Run everything with 80/20 split
split = 0.8
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)

d[f"KNN_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train, y_test)
d[f"KNN_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train, y_test)
d[f"SVC_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train, y_test)
d[f"SVC_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train, y_test)
d[f"LR_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train, y_test)
d[f"LR_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train, y_test)

##Run everything with 90/10 split
split = 0.9
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)

d[f"KNN_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train, y_test)
d[f"KNN_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train, y_test)
d[f"SVC_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train, y_test)
d[f"SVC_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train, y_test)
d[f"LR_PCA_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train, y_test)
d[f"LR_Feat_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train, y_test)

#Save data of part 1
df_1 = pd.DataFrame(data =d, index = ['Train', 'Cross', 'Test', 'Opt_Feat_or_PCA'])
df_1.to_csv('./data.csv', sep=" ")

##Part 2 Theme 1 mislabeling

def mislabel(mislabel_fraction, y_train):
    labels = set(labels_df["x"])

    num_samples = len(y_train)
    num_mislabels = int(mislabel_fraction * num_samples)
    mislabel_indices = np.random.choice(num_samples, num_mislabels, replace=False)

    y_train_noise = y_train.copy()

    for i in mislabel_indices:
        correct = y_train[i]
        y_train_noise[i] = np.random.choice(list(labels - set([correct])))
    
    return y_train_noise

##Creating dictionary to save data of part 2
d = dict()

##Mislabel fraction 0.2, 70/30 split
mislabel_fraction = 0.2
split = 0.7
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.2, 80/20 split
mislabel_fraction = 0.2
split = 0.8
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)


##Mislabel fraction 0.2, 90/10 split
mislabel_fraction = 0.2
split = 0.9
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.5, 70/30 split
mislabel_fraction = 0.5
split = 0.7
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.5, 80/20 split
mislabel_fraction = 0.5
split = 0.8
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.5, 90/10 split
mislabel_fraction = 0.5
split = 0.9
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.9, 70/30 split
mislabel_fraction = 0.9
split = 0.7
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.9, 80/20 split
mislabel_fraction = 0.9
split = 0.8
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

##Mislabel fraction 0.9, 90/10 split
mislabel_fraction = 0.9
split = 0.9
X_train, X_test, y_train, y_test = pre_process(df, labels_df, split)
y_train_noise = mislabel(mislabel_fraction, y_train)

d[f"KNN_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_PCA(X_train, X_test, y_train_noise, y_test)
d[f"KNN_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = KNN_features(X_train, X_test, y_train_noise, y_test)
d[f"SVC_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_PCA(X_train, X_test, y_train_noise, y_test)
d[f"SVC_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = SVC_features(X_train, X_test, y_train_noise, y_test)
d[f"LR_PCA_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_PCA(X_train, X_test, y_train_noise, y_test)
d[f"LR_Feat_miss_{mislabel_fraction:.1f}_{(1-split)*100:.0f}_{split*100:.0f}"] = LR_features(X_train, X_test, y_train_noise, y_test)

#Save data of part 2
df_miss = pd.DataFrame(data =d, index = ['Train', 'Cross', 'Test', 'Opt_Feat_or_PCA'])
df_miss.to_csv('./data_miss.csv', sep=" ")

##Print data from both parts
td = pd.read_csv('./data.csv', sep = " ", header=0, index_col=0)
print(td)
td_miss= pd.read_csv('./data_miss.csv', sep = " ", header=0, index_col=0)
print(td_miss)

