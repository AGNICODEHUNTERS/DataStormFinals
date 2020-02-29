import pandas as pd


test = pd.read_csv('test.csv')
test = test.groupby(['customer_id','year','gender','month']).sum()

test.to_csv('tttt.csv')
