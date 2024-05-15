import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../Data/CATSnDOGS.csv', sep="," ,header = 0)
#print(df.columns)

kb_scores = np.zeros([4096])
print(kb_scores)

for n in range(10):
    kb = np.array(pd.read_csv('data_1b_kb_' + str(n), sep = ' ')['0'])
    kb_scores += kb
    print(kb_scores)