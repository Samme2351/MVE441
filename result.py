import pandas as pd

data1 = pd.read_csv('data.csv', sep=" " ,header=0)
data2 = pd.read_csv('run1/data.csv', sep=" " ,header=0)
data3 = pd.read_csv('run2/data.csv', sep=" " ,header=0)
data4 = pd.read_csv('run3/data.csv', sep=" " ,header=0)
data5 = pd.read_csv('run4/data.csv', sep=" " ,header=0)
print(data1)
print(data2)
data = (data1+data2+data3+data4+data5).div(5)

data.to_csv('./mean.csv', sep=" ")