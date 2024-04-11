import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
#from tqdm import tqdm

scaler = preprocessing.StandardScaler()

df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)
labels_df = pd.read_csv('Data/TCGAlabels', sep=" " ,header=0)


#print(df)
#print(labels_df)

# Sepearate into test data and working data
n_rows = len(df) 
m_test = 900

test_data = df.head(m_test)
working_data = df.tail(n_rows-m_test)
scaled_work_data = pd.DataFrame(scaler.fit_transform(working_data.transpose()))

work_labels = labels_df.tail(n_rows-m_test)
#print(work_labels)
#print(test_data)
print(scaled_work_data)
#Standardizing the rows (transposing as fit_transform standardizes along columns)?
# Use PCA and cross-validation

num_components_range = range(1, 25)  # Example range, adjust as needed

# Initialize lists to store mean cross-validation scores for each number of components
SVC_mean_scores = []
KNN_mean_scores = []

# Loop over different numbers of components
for n_components in num_components_range:
    # Create a pipeline for classifiers
    SVC_pipeline = pipeline.make_pipeline(PCA(n_components=n_components), SVC())  # Example pipeline, replace with your choice
    KNN_pipeline = pipeline.make_pipeline(PCA(n_components=n_components), KNeighborsClassifier())
    
    # Mean cross-validation scores:
    SVC_scores = cross_val_score(SVC_pipeline, scaled_work_data, work_labels.values.ravel(), cv=5)
    SVC_mean_score = SVC_scores.mean()
    
    KNN_scores = cross_val_score(KNN_pipeline, scaled_work_data, work_labels.values.ravel(), cv=5)
    KNN_mean_score = KNN_scores.mean()
    
    SVC_mean_scores.append(SVC_mean_score)
    KNN_mean_scores.append(KNN_mean_score)

# Find the optimal number of components based on the highest mean cross-validation score
SVC_optimal_n_components = num_components_range[SVC_mean_scores.index(max(SVC_mean_scores))]
KNN_optimal_n_components = num_components_range[KNN_mean_scores.index(max(KNN_mean_scores))]

# Print the optimal number of components
print("Optimal number of components:", SVC_optimal_n_components)
print("Optimal number of components:", KNN_optimal_n_components)

