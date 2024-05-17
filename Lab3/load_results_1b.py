import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists

df = pd.read_csv('./Data/CATSnDOGS.csv', sep="," ,header = 0)
#print(list(df.columns))


## LR scores: absolute value of coefficients
lr_scores = np.zeros([4096])

n = 0
file_exists = True
LR_repeating = set()
while file_exists:
    if exists('data_1b_lr_' + str(n)):
        lr_result_temp = abs(np.array(pd.read_csv('data_1b_lr_' + str(n), sep = " ")['0']))
        lr_scores += lr_result_temp
        n += 1
        if len(LR_repeating) == 0:
            LR_repeating = set([x for _, x in sorted(zip(lr_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)][:100])
        else:
            LR_repeating = LR_repeating & set([x for _, x in sorted(zip(lr_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)][:100])
        #print(lr_scores)
    else:
        file_exists = False
print("repeting")
print(LR_repeating)
print("stop \n\n")

# Sort LR scores
LR_imp_features = [x for _, x in sorted(zip(lr_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(LR_imp_features[:100])


## KB scores: f_classif scores
kb_scores = np.zeros([4096])

n = 0
file_exists = True
kb_repeating = set()
while file_exists:
    if exists('data_1b_lr_' + str(n)):
        kb_result_temp = np.array(pd.read_csv('data_1b_kb_' + str(n), sep = ' ')['0'])
        kb_scores += kb_result_temp
        n += 1
        if len(kb_repeating) == 0:
            kb_repeating = set([x for _, x in sorted(zip(kb_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)][:100])
        else:
            kb_repeating = kb_repeating & set([x for _, x in sorted(zip(kb_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)][:100])
        #print(kb_scores)
    else:
        file_exists = False

print("repeting")
print(kb_repeating)
print("stop \n\n")

# Sort KB scores
KB_imp_features = [x for _, x in sorted(zip(kb_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(KB_imp_features[:100])


## NC scores: absolute distance between centroid and overall centroid
nc_scores = np.zeros([4096])

n = 0
file_exists = True
nc_repeating = set()
while file_exists:
    if exists('data_1b_lr_' + str(n)):
        nc_result_temp = pd.read_csv('data_1b_nc_' + str(n), sep = " ")
        centroid_1 = np.array(nc_result_temp.loc[0])
        centroid_1 = np.delete(centroid_1, 0)
        centroid_2 = np.array(nc_result_temp.loc[1])
        centroid_2 = np.delete(centroid_2, 0)
        nc_scores += abs(centroid_1 - (centroid_1 + centroid_2)/2)
        n += 1
        if len(nc_repeating) == 0:
            nc_repeating = set([x for _, x in sorted(zip(nc_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)][:100])
        else:
            nc_repeating = nc_repeating & set([x for _, x in sorted(zip(nc_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)][:100])
        #print(nc_scores.max())
    else:
        file_exists = False
print("repeting")
print(nc_repeating)
print("stop \n\n")

print("Repetas in all runs for all models")
print(nc_repeating & LR_repeating & kb_repeating)
print("\n")
# Sort NC scores
NC_imp_features = [x for _, x in sorted(zip(nc_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(NC_imp_features[:100])

print(set(LR_imp_features[:200]) & set(KB_imp_features[:200]) & set(NC_imp_features[:200]))