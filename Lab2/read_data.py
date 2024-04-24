import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

labels = pd.read_csv("data.csv", sep=" " ,header=0, index_col=0)
data = pd.read_csv("data.csv", sep=" " ,header=0, index_col=0)
#print(data["Noise_0.0"][5][1:-1].split(", "))

label_dic = dict()

for i in [0.1, 0.5, 1, 3]:
    labels = data[f"Noise_{i:.1f}"][4][2:-2].split("', '")
    for label in labels:
        if label in label_dic:
            label_dic[label] +=1
        else:
            label_dic[label] = 1


for i in [0, 0.1, 0.5, 1, 3]:
    values = [float(val) for val in data[f"Noise_{i:.1f}"][5][1:-1].split(", ")]
    #print(values)
    print(sum([val>float('1.0e-04') for val in values]))

#print(sum([label_dic[label] == 4 for label in label_dic]))






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