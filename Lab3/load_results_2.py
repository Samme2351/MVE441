import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists

# Function for averaging results of runs
def average(split_str):
    file_exists = True
    ov = np.zeros([8])
    n = 0

    while file_exists:
        if exists('data_2_' + split_str + '_' + str(n)):
            d = pd.read_csv('data_2_' + split_str + '_' + str(n), sep=" ")
            d = np.array(d['overall'])
            ov += d
            n += 1
        else:
            file_exists = False

    return ov/n


# Average results for lowest split
low_split = str(0.05)    # Change this if split is changed
ov_lo = average(low_split)
print(ov_lo)

# Average results for middle split
mid_split = str(0.4)    # Change this if split is changed
ov_mid = average(mid_split)
print(ov_mid)

# Average results for highest split
hi_split = str(0.7)     # Change this if split is changed
ov_hi = average(hi_split)
print(ov_hi)

# Order of results: KNN, LDA, QDA, LR, Lasso, SVC, RF, XGB