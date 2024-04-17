import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

labels = pd.read_csv('data_1.csv', sep=" " ,header=0)
data1 = pd.read_csv('data_1.csv', sep=" " ,header=0).to_numpy()
data2 = pd.read_csv('data_2.csv', sep=" " ,header=0).to_numpy()
data3 = pd.read_csv('data_3.csv', sep=" " ,header=0).to_numpy()
data4 = pd.read_csv('data_4.csv', sep=" " ,header=0).to_numpy()
data5 = pd.read_csv('data_5.csv', sep=" " ,header=0).to_numpy()
#data6 = pd.read_csv('data_6.csv', sep=" " ,header=0).to_numpy()
#data7 = pd.read_csv('data_7.csv', sep=" " ,header=0).to_numpy()

data = (data1+data2+data3+data4+data5)/5#+data6+data7)/7

x = [0.7, 0.8, 0.9]
fig, axs = plt.subplots(2,3)
s = 0
color = ["blue", "green", "red", "purple", "pink", "black"]
for j in range(3):
    for k in range(2):
        for i in range(3):
            axs[k,j].plot(x, [data[i][s], data[i][6+s], data[i][12+s]], color = color[i])
            axs[k,j].set_title(labels.columns[s][:-6])
        s += 1

for ax in axs.flat:
    ax.set(xlabel='Precentage of data used for training', ylabel='Error')

for ax in axs.flat:
    ax.label_outer()


location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Training error", "Cross-val error", "Test error"], loc=0, frameon=legend_drawn_flag)
plt.show()



'''
x = [0.7, 0.8, 0.9]
color = ["blue", "green", "red", "purple", "pink", "black"]
for i in range(6):
    plt.plot(x, [data.iloc[0][i], data.iloc[0][6+i], data.iloc[0][12+i]], '*-',label = data.columns[i][:-6], color = color[i])
    #plt.plot(x, [data.iloc[1][i], data.iloc[1][6+i], data.iloc[1][12+i]],'-' , label = data.columns[i][:-6], color = color[i])
    plt.plot(x, [data.iloc[2][i], data.iloc[2][6+i], data.iloc[2][12+i]],'--' ,label = data.columns[i][:-6],color = color[i])
plt.legend()
plt.show()
'''

#data.to_csv('./mean.csv', sep=" ")

#Misses

labels = pd.read_csv('data_miss.csv', sep=" " ,header=0, index_col=0)
data1 = pd.read_csv('data_miss_1.csv', sep=" " ,header=0, index_col=0).to_numpy()
data2 = pd.read_csv('data_miss_2.csv', sep=" " ,header=0, index_col=0).to_numpy()
data3 = pd.read_csv('data_miss_3.csv', sep=" " ,header=0, index_col=0).to_numpy()
data4 = pd.read_csv('data_miss_4.csv', sep=" " ,header=0, index_col=0).to_numpy()
data5 = pd.read_csv('data_miss_5.csv', sep=" " ,header=0, index_col=0).to_numpy()
data6 = pd.read_csv('data_miss_6.csv', sep=" " ,header=0, index_col=0).to_numpy()
data7 = pd.read_csv('data_miss_7.csv', sep=" " ,header=0, index_col=0).to_numpy()

data = (data1+data2+data3+data4+data5+data6+data7)/7


x = [0.7, 0.8, 0.9]
fig, axs = plt.subplots(6,3)

type = ['-', '*-', '--']
color = ["blue", "green", "red", "purple", "pink", "black"]
s= 0
for k in range(3):
    for j in range(2):
        for i in range(3):
            axs[j,k].plot(x, [data[i][s], data[i][6+s], data[i][12+s]], color = color[i])
            axs[j,k].set_title(lab.columns[s][:-6])

            axs[2+j,k].plot(x, [data[i][18+s], data[i][24+s], data[i][30+s]], color = color[i])
            axs[2+j,k].set_title(lab.columns[18+s][:-6])

            axs[4+j,k].plot(x, [data[i][36+s], data[i][42+s], data[i][48+s]], color = color[i])
            axs[4+j,k].set_title(lab.columns[36+s][:-6])
        s += 1

for ax in axs.flat:
    ax.set(xlabel='Precentage of data used for training', ylabel='Error')

for ax in axs.flat:
    ax.label_outer()


location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Training error", "Cross-val error", "Test error"], loc=0, frameon=legend_drawn_flag)
plt.show()
