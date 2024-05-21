import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
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
        lr_result_temp = np.abs(np.array(pd.read_csv('data_1b_lr_' + str(n), sep = " ")['0']))

        lr_scores += lr_result_temp
        n += 1
        
        lr_sort_list = sorted(zip(lr_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)
        lr_sort_list =[elem for elem in lr_sort_list if elem[0] != 0] 

        if len(LR_repeating) == 0:
            LR_repeating = set([x for _, x in lr_sort_list][:400])
        else:
            LR_repeating = LR_repeating & set([x for _, x in lr_sort_list][:400])
        #print(lr_scores)
    else:
        file_exists = False
print("repeting")
print(LR_repeating)
print("stop \n\n")

# Sort LR scores
LR_imp_features = [x for _, x in sorted(zip(lr_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(LR_imp_features[:100])




perm_scores = np.zeros([4096])

n = 0
file_exists = True
perm_repeating = set()
while file_exists:
    if exists('data_1b_perm_' + str(n)):
        perm_result_temp = np.abs(np.array(pd.read_csv('data_1b_perm_' + str(n), sep = " ")['0']))
        #print(lr_result_temp)
        perm_scores += perm_result_temp
        n += 1

        perm_sort_list = sorted(zip(perm_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)
        perm_sort_list =[elem for elem in perm_sort_list if elem[0] != 0] 

        if len(perm_repeating) == 0:
            perm_repeating = set([x for _, x in perm_sort_list][:400])
        else:
            perm_repeating = perm_repeating & set([x for _, x in perm_sort_list][:400])
        #print(perm_scores)
    else:
        file_exists = False
print("repeting")
print(perm_repeating)
print("stop \n\n")


# Sort LR scores
perm_imp_features = [x for _, x in sorted(zip(perm_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(LR_imp_features[:100])
## KB scores: f_classif scores
kb_scores = np.zeros([4096])

n = 0
file_exists = True
kb_repeating = set()
while file_exists:
    if exists('data_1b_kb_' + str(n)):
        kb_result_temp = np.abs(np.array(pd.read_csv('data_1b_kb_' + str(n), sep = ' ')['0']))
        kb_scores += kb_result_temp
        n += 1

        kb_sort_list = sorted(zip(kb_result_temp, list(df.columns)), key=lambda pair: pair[0], reverse=True)
        kb_sort_list =[elem for elem in kb_sort_list if elem[0] != 0] 

        if len(kb_repeating) == 0:
            kb_repeating = set([x for _, x in kb_sort_list][:400])
        else:
            kb_repeating = kb_repeating & set([x for _, x in kb_sort_list][:400])
        #print(kb_scores)
    else:
        file_exists = False

print("repeting")
print(kb_repeating)
print("stop \n\n")

# Sort KB scores
kb_imp_features = [x for _, x in sorted(zip(kb_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]

#print(KB_imp_features[:100])


## NC scores: absolute distance between centroid and overall centroid
nc_scores = np.zeros([4096])

n = 0
file_exists = True
nc_repeating = set()
while file_exists:
    if exists('data_1b_nc_' + str(n)):
        nc_result_temp = pd.read_csv('data_1b_nc_' + str(n), sep = " ")
        centroid_1 = np.array(nc_result_temp.loc[0])
        centroid_1 = np.delete(centroid_1, 0)
        centroid_2 = np.array(nc_result_temp.loc[1])
        centroid_2 = np.delete(centroid_2, 0)
        centroid_new = abs(centroid_1 - (centroid_1 + centroid_2)/2)
        nc_scores += centroid_new
        n += 1

        nc_sort_list = sorted(zip(centroid_new, list(df.columns)), key=lambda pair: pair[0], reverse=True)
        nc_sort_list =[elem for elem in nc_sort_list if elem[0] != 0] 

        if len(nc_repeating) == 0:
            nc_repeating = set([x for _, x in nc_sort_list][:400])
        else:
            nc_repeating = nc_repeating & set([x for _, x in nc_sort_list][:400])
        #print(nc_scores.max())
    else:
        file_exists = False
print("repeting")
print(nc_repeating)
print("stop \n\n")

# Sort NC scores
nc_imp_features = [x for _, x in sorted(zip(nc_scores, list(df.columns)), key=lambda pair: pair[0], reverse=True)]
#print(NC_imp_features[:100])


print("Repetas in all runs for all models")
print(nc_repeating & LR_repeating & kb_repeating)
print("\n")


common = set(LR_imp_features[:400]) & set(kb_imp_features[:400]) & set(nc_imp_features[:400])




imp_feat = set(LR_imp_features[:400])
imp_feat = [int(name.replace('V', '')) for name in imp_feat]

imp_plt = df.iloc[20].to_numpy().astype(float)

cmap = colormaps['bone']
cmap.set_bad((1,0,0,1))
imp_plt[imp_feat] = np.nan
imp_plt = imp_plt.reshape((64,64)).transpose()
plt.imshow(imp_plt, cmap = cmap)
plt.title("The most important features for LR (Beta)")
plt.show()



imp_feat = LR_repeating
imp_feat = [int(name.replace('V', '')) for name in imp_feat]

imp_plt = df.iloc[20].to_numpy().astype(float)

cmap = colormaps['bone']
cmap.set_bad((1,0,0,1))
imp_plt[imp_feat] = np.nan
imp_plt = imp_plt.reshape((64,64)).transpose()
plt.imshow(imp_plt, cmap = cmap)
plt.title("The features which are important in every run for LR (Beta)")
plt.show()