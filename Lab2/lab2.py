import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm



def test_split(split, df, labels_df):
    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-split, random_state=42)
    
    return X_train, X_test, y_train, y_test

def main():
    split = 0.7
    df = pd.read_csv('data/TCGAdata.txt', sep = " ", header = 0)
    labels_df = pd.read_csv('data/TCGALabels', sep = " ", header = 0)

    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-split, random_state=42)
    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)

    train_pred = forest.predict(X_train)
    train_err = 1- accuracy_score(y_train, train_pred)

    test_pred = forest.predict(X_test)
    test_err = 1- accuracy_score(y_test, test_pred)

    print("train error: ", train_err)
    print("test error: ", test_err)

    print(forest.feature_importances_)
    print(sum(forest.feature_importances_))

    return 0

main()