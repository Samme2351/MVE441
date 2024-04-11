import pandas as pd
from tabulate import tabulate


d = {'Log': [0,1,None], 'SVC': [1,5,3]}
df = pd.DataFrame(data =d, columns = ['Log', 'SVC'], index = ['Train', 'Test', 'Cross'])

df.to_csv('./data.csv', sep=" ")


td = pd.DataFrame()
td= pd.read_csv('./data.csv', sep = " ", header=0, index_col=0)
print(td)
#print(tabulate(td, headers = 'keys', tablefmt = 'psql'))