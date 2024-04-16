import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('data.csv', sep=" " ,header=0)
data2 = pd.read_csv('run1/data.csv', sep=" " ,header=0)
data3 = pd.read_csv('run2/data.csv', sep=" " ,header=0)
data4 = pd.read_csv('run3/data.csv', sep=" " ,header=0)
data5 = pd.read_csv('run4/data.csv', sep=" " ,header=0)
data6 = pd.read_csv('data_1.csv', sep=" " ,header=0)
data7 = pd.read_csv('data_2.csv', sep=" " ,header=0)

print(data1)
print(data2)
data = (data1+data2+data3+data4+data5+data6+data7).div(7)
print(data)
print(data.iloc[0])

x = [0.7, 0.8, 0.9]
#KNN_PCA_70_30 PCA["KNN_PCA_30_70"]
plt.plot(x, data.iloc[1])
plt.show()

#data.to_csv('./mean.csv', sep=" ")