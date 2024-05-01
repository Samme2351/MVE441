import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dat = 20

data = []
data_gb = []
data_cat = []
data_cat_gb = []
for i in range(dat):
    labels = pd.read_csv("./Results/data_0.csv", sep=" " ,header=0, index_col=0)
    data.append(pd.read_csv(f"./Results/data_{i}.csv", sep=" " ,header=0, index_col=0)) 
    data_gb.append(pd.read_csv(f"./Results/data_gb_{i}.csv", sep=" " ,header=0, index_col=0))
    data_cat.append(pd.read_csv(f"./Results/data_cat_{i}.csv", sep=" " ,header=0, index_col=0))
    data_cat_gb.append(pd.read_csv(f"./Results/data_gb_cat_{i}.csv", sep=" " ,header=0, index_col=0))

df = pd.read_csv('./data/TCGAdata.txt', sep=" " ,header=0, index_col= 0)
df_cat = pd.read_csv('./data/CATSnDOGS.csv', sep="," ,header=0)

## Class errors
color = ["blue", "green", "red", "purple", "pink", "black"]
test = [[],[],[],[],[],[]]
x = [0.0, 0.1, 0.5, 1.0, 3.0]
for noise in x:
    dic = dict()
    for i in range(dat):
        for j in data_gb[i].iloc[5][f"Noise_{noise:.1f}"].split(","): #data or data_gb
            clas = j.split(': ')
            try:
                flo = float(clas[1])
            except:
                flo = float(clas[1][:-1])
            if clas[0][2:-1] not in dic:
                dic[clas[0][2:-1]] = flo
            else:
                dic[clas[0][2:-1]] += flo

    for clas in dic:
        dic[clas] = dic[clas]/20
    print(f"For noise level {noise}", dic)

    for i in range(len(dic)):
        classes = list(dic)
        test[i].append(dic[classes[i]])

for i in range(6):
    plt.plot(x,test[i], color = color[i])
    plt.title("XGBoost")

plt.xlabel('Standard deviation of noise')
plt.ylabel('Error')
plt.ylim([0,1])
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(list(dic), loc=0, frameon=legend_drawn_flag)
plt.show()




## Feature importance
amount = 0
freq = 0
cats = False

for noise in [0.0, 0.1, 0.5, 1.0, 3.0]:
    if cats == True:
        importance = pd.DataFrame(np.zeros(4096), df_cat.columns).T # Total value of importance
        imp_labels = pd.DataFrame(np.zeros(4096), df_cat.columns).T # How often Each feature had non-zero importance
        
    else:
        importance = pd.DataFrame(np.zeros(2000), df.columns).T # Total value of importance
        imp_labels = pd.DataFrame(np.zeros(2000), df.columns).T # How often Each feature had non-zero importance
    
    for d in data_gb: #Change to data or data_gb
        labels = d[f"Noise_{noise:.1f}"][6][2:-2].split("', '")
        values = [float(val) for val in d[f"Noise_{noise:.1f}"][7][1:-1].split(", ")]
        imp_labels.loc[:,labels] += np.ones(len(labels))
        importance.loc[:, labels] += values
    mean_imp = importance/dat



    
    if amount == True:
        print(f"For error level {noise:.1f} we have")
        for i in range(0,21):
            print(sum(imp_labels.iloc[0]==i), f" features appeared {i} times")

    if freq == True:
        x = list(range(1,2001))
        plt.plot(x, mean_imp.iloc[0])
        plt.show()




#plot error vs noise level

std = [[], [], [], [], [], [], [], [], [], []]
std_gb = [[], [], [], [], [], [], [], [], [], []]
for i in range(dat):
    s = 0
    for j in range(2):
        for k in range(5):
            std[s].append(data[i].iloc[1:3].astype(float).to_numpy()[j][k])
            std_gb[s].append(data_gb[i].iloc[1:3].astype(float).to_numpy()[j][k])
            s += 1

for i in range(10):
    std[i] = np.array(std[i]).std()
    std_gb[i] = np.array(std_gb[i]).std()


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
    if i == 0:
        axs[0].plot(x, errors.iloc[i].to_numpy()) 
        axs[1].plot(x, errors_gb.iloc[i].to_numpy()) 
    else: 
        axs[0].errorbar(x, errors.iloc[i].to_numpy(), std[(i-1)*5:(i)*5]) 
        axs[1].errorbar(x, errors_gb.iloc[i].to_numpy(), std_gb[(i-1)*5:(i)*5]) 
axs[0].set_title("Random Forest")
axs[1].set_title("XGBoost")

for ax in axs.flat:
    ax.set(xlabel='Standard deviation of noise', ylabel='Error',ylim =[0,0.45] )
location = 0 # For the best location
legend_drawn_flag = True
plt.legend(["Training error", "Cross-val error", "Test error"], loc=0, frameon=legend_drawn_flag)
#plt.show()
