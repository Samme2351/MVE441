import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


dat = 20
data = []
data_gb = []
data_cat =[]
for i in range(dat):
    labels = pd.read_csv("./Results/data_0.csv", sep=" " ,header=0, index_col=0)
    data.append(pd.read_csv(f"./Results/data_{i}.csv", sep=" " ,header=0, index_col=0)) 
    data_gb.append(pd.read_csv(f"./Results/data_gb_{i}.csv", sep=" " ,header=0, index_col=0))
    data_cat.append(pd.read_csv(f"./Results/data_cat_{i}.csv", sep=" " ,header=0, index_col=0))
#print(data["Noise_0.0"][5][1:-1].split(", "))

df = pd.read_csv('./data/TCGAdata.txt', sep=" " ,header=0, index_col= 0)


## Feature importance
for noise in [0.0, 0.1, 0.5, 1.0, 3.0]:
    importance = pd.DataFrame(np.zeros(2000), df.columns).T # Total value of importance
    imp_labels = pd.DataFrame(np.zeros(2000), df.columns).T # How often Each feature had non-zero importance
    for d in data_gb: #Change to data or data_gb
        labels = d[f"Noise_{noise:.1f}"][6][2:-2].split("', '")
        values = [float(val) for val in d[f"Noise_{noise:.1f}"][7][1:-1].split(", ")]
        imp_labels.loc[:,labels] += np.ones(len(labels))
        importance.loc[:, labels] += values
            
    print(imp_labels)
    print(importance)
    print(f"For error level {noise:.1f} we have")
    for i in range(0,21):
        print(sum(imp_labels.iloc[0]==i), f" features appeared {i} times")




#plot error vs noise level
errors = data[0].iloc[0:3].astype(float)
errors_gb = data_gb[0].iloc[0:3].astype(float)
for i in range(1,dat):
    errors += data[i].iloc[0:3].astype(float)
    errors_gb += data_gb[i].iloc[0:3].astype(float)

errors = errors/dat
errors_gb = errors_gb/dat

x = [0.0, 0.1, 0.5, 1.0, 3.0]
fig, axs = plt.subplots(1,2)
s = 0
color = ["blue", "green", "red", "purple", "pink", "black"]

for i in range(3):
    axs[0].plot(x, errors.iloc[i].to_numpy()) 
    axs[1].plot(x, errors_gb.iloc[i].to_numpy()) 
axs[0].set_title("Random Forest")
axs[1].set_title("XGBoost")

for ax in axs.flat:
    ax.set(xlabel='Standard deviation of noise', ylabel='Error',ylim =[0,0.45] )
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Training error", "Cross-val error", "Test error"], loc=0, frameon=legend_drawn_flag)
#plt.show()






'''
label_dic = dict()

for i in [0.0, 0.1, 0.5, 1, 3]:
    labels = data[f"Noise_{i:.1f}"][4][2:-2].split("', '")
    for label in labels:
        if label in label_dic:
            label_dic[label] +=1
        else:
            label_dic[label] = 1


for i in [0, 0.1, 0.5, 1, 3]:
    values = [float(val) for val in data[f"Noise_{i:.1f}"][7][1:-1].split(", ")]
    #print(values)
    print(sum([val>float('1.0e-03') for val in values]))

#print(sum([label_dic[label] == 4 for label in label_dic]))
'''





'''
data2 = pd.read_csv('data_2.csv', sep=" " ,header=0, index_col=0).to_numpy()
data3 = pd.read_csv('data_3.csv', sep=" " ,header=0, index_col=0).to_numpy()
data4 = pd.read_csv('data_4.csv', sep=" " ,header=0, index_col=0).to_numpy()
data5 = pd.read_csv('data_5.csv', sep=" " ,header=0, index_col=0).to_numpy()
data6 = pd.read_csv('data_6.csv', sep=" " ,header=0, index_col=0).to_numpy()
data7 = pd.read_csv('data_7.csv', sep=" " ,header=0, index_col=0).to_numpy()
data8 = pd.read_csv('data_8.csv', sep=" " ,header=0, index_col=0).to_numpy()
data9 = pd.read_csv('data_9.csv', sep=" " ,header=0, index_col=0).to_numpy()
data10 = pd.read_csv('data_10.csv', sep=" " ,header=0, index_col=0).to_numpy()
data11 = pd.read_csv('data_11.csv', sep=" " ,header=0, index_col=0).to_numpy()

data = (data1+data2+data3+data4+data5+data6+data7+data8+data9+data10+data11)/11
'''
'''
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
    ax.set(xlabel='Precentage of data used for training', ylabel='Error',ylim =[0,0.012] )



location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Training error", "Cross-val error", "Test error"], loc=0, frameon=legend_drawn_flag)
plt.show()



#Misses

labels = pd.read_csv('data_miss_1.csv', sep=" " ,header=0, index_col=0)
data1 = pd.read_csv('data_miss_1.csv', sep=" " ,header=0, index_col=0).to_numpy()
data2 = pd.read_csv('data_miss_2.csv', sep=" " ,header=0, index_col=0).to_numpy()
data3 = pd.read_csv('data_miss_3.csv', sep=" " ,header=0, index_col=0).to_numpy()
data4 = pd.read_csv('data_miss_4.csv', sep=" " ,header=0, index_col=0).to_numpy()
data5 = pd.read_csv('data_miss_5.csv', sep=" " ,header=0, index_col=0).to_numpy()
data6 = pd.read_csv('data_miss_6.csv', sep=" " ,header=0, index_col=0).to_numpy()
data7 = pd.read_csv('data_miss_7.csv', sep=" " ,header=0, index_col=0).to_numpy()
data8 = pd.read_csv('data_miss_8.csv', sep=" " ,header=0, index_col=0).to_numpy()
data9 = pd.read_csv('data_miss_9.csv', sep=" " ,header=0, index_col=0).to_numpy()
data10 = pd.read_csv('data_miss_10.csv', sep=" " ,header=0, index_col=0).to_numpy()
data11 = pd.read_csv('data_miss_11.csv', sep=" " ,header=0, index_col=0).to_numpy()

data = (data1+data2+data3+data4+data5+data6+data7+data8+data9+data10+data11)/11


def plot(version): #version must be 0 (20% mislabeling), 1 (50% mislabeling) or 2 (90% mislabeling)
    x = [0.7, 0.8, 0.9]
    fig, axs = plt.subplots(2,3)

    color = ["blue", "green", "red"]
    s= 0
    for k in range(3):
        for j in range(2):
            for i in range(3):
                axs[j,k].plot(x, [data[i][s+18*version], data[i][6+18*version+s], data[i][12+18*version + s]], color = color[i])
                axs[j,k].set_title(labels.columns[s+18*version][:-6])
            s += 1
            
    y = [0.25, 0.65,1]

    for ax in axs.flat:
        ax.set(xlabel='Precentage of data used for training', ylabel='Error',ylim =[0,y[version]])

    location = 0 # For the best location
    legend_drawn_flag = True
    plt.legend(["Training error", "Cross-val error", "Test error"], loc=0, frameon=legend_drawn_flag)
    plt.show()


plot(0)
plot(1)
plot(2)
'''