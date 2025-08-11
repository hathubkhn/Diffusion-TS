import pandas as pd

df = pd.read_csv('data/short_range/JNJ.csv')
# delete "Symbol", "industry","date" column
df = df.drop(columns=['Date', 'Symbol'])
# df = df.drop(columns=['industry'])
# df = df.drop(columns=['Symbol'])

df.to_csv('data/short_range/JNJ.csv', index=False)
