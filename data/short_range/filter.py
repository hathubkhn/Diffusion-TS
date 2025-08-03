import pandas as pd

df = pd.read_csv('/home/user11/binhnkt/data_new/XOM.csv')
print(df.head())
# delete "Symbol", "industry","date" column
df = df.drop(columns=['Date', 'Symbol'])
# df = df.drop(columns=['industry'])
# df = df.drop(columns=['Symbol'])

df.to_csv('data/short_range/XOM.csv', index=False)
