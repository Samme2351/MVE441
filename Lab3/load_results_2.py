import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ov = np.zeros([7])
for n in range(10):
    d = pd.read_csv('data_2_0.7_' + str(n), sep=" ")
    d = np.array(d['overall'])
    ov += d

print(ov/10)