import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../Data/CATSnDOGS.csv', sep="," ,header = 0)
#print(list(df.columns))


## LR scores: absolute value of coefficients
lr_scores = np.zeros([4096])

for n in range(10):
    lr_result_temp = abs(np.array(pd.read_csv('data_1b_lr_' + str(n), sep = " ")['0']))
    lr_scores += lr_result_temp
    #print(lr_scores)

# Sort LR scores
LR_imp_features = [x for _, x in sorted(zip(lr_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(LR_imp_features[:100])


## KB scores: f_classif scores
kb_scores = np.zeros([4096])

for n in range(10):
    kb_result_temp = np.array(pd.read_csv('data_1b_kb_' + str(n), sep = ' ')['0'])
    kb_scores += kb_result_temp
    #print(kb_scores)

# Sort KB scores
KB_imp_features = [x for _, x in sorted(zip(kb_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(KB_imp_features[:100])


## NC scores: absolute distance between centroid and overall centroid
nc_scores = np.zeros([4096])

for n in range(10):
    nc_result_temp = pd.read_csv('data_1b_nc_' + str(n), sep = " ")
    centroid_1 = np.array(nc_result_temp.loc[0])
    centroid_1 = np.delete(centroid_1, 0)
    centroid_2 = np.array(nc_result_temp.loc[1])
    centroid_2 = np.delete(centroid_2, 0)
    nc_scores += abs(centroid_1 - (centroid_1 + centroid_2)/2)
    #print(nc_scores.max())

# Sort NC scores
NC_imp_features = [x for _, x in sorted(zip(nc_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(NC_imp_features[:100])

print(set(LR_imp_features[:200]) & set(KB_imp_features[:200]) & set(NC_imp_features[:200]))