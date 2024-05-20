import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists

# Function for averaging results of runs
def average(split_str):
    file_exists = True
    ov = np.zeros([10])
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

methods = ("KNN", "LDA", "QDA", "LR", "Lasso", "SVC lin", "SVC poly", "RF", "XGB", "NN")
results = {
    'KNN': (ov_lo[0], ov_mid[0], ov_hi[0]),
    'LDA': (ov_lo[1], ov_mid[1], ov_hi[1]),
    'QDA': (ov_lo[2], ov_mid[2], ov_hi[2]),
    'LR': (ov_lo[3], ov_mid[3], ov_hi[3]),
    'Lasso': (ov_lo[4], ov_mid[4], ov_hi[4]),
    'SVC lin': (ov_lo[5], ov_mid[5], ov_hi[5]),
    'SVC poly': (ov_lo[6], ov_mid[6], ov_hi[6]),
    'RF': (ov_lo[7], ov_mid[7], ov_hi[7]),
    'XGB': (ov_lo[8], ov_mid[8], ov_hi[8]),
    'NN': (ov_lo[9], ov_mid[9], ov_hi[9])
}

df = pd.DataFrame({'Test error, 0.05/0.95 split': ov_lo,
                   'Test error, 0.4/0.6 split': ov_mid,
                   'Test error, 0.7/0.3 split': ov_hi}, index=methods)

ax = df.plot.bar(rot=0)
ax.set_ylim([0.3, 1.0])

ax.set_title('Test accuracy for different methods')

plt.show()