import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Data/TCGAdata.txt', sep=" " ,header=0)

#scaler = StandardScaler()
#df= pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df['V84'])