import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

def main():
    split = 0.7
    df = pd.read_csv('data/TCGAdata.txt', sep = " ", header = 0)
    labels_df = pd.read_csv('data/TCGALabels', sep = " ", header = 0)
    #df = pd.read_csv('data/CATSnDOGS.csv', sep = ",", header = 0)
    #labels_df = pd.read_csv('data/Labels.csv', sep = ",", header = 0)

    X_train, X_test, y_train, y_test = train_test_split(df, labels_df.values.ravel(), test_size=1-split, random_state=42)
    
    # TODO: Noise, depth, n trees, learnrate, subsample, split, leaf, sample size.
    noises = [0, 0.2, 0.4, 0.8, 1.6, 3.2]

    return 0

main()