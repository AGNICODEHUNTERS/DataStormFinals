import pandas as pd

test_detail = pd.read_csv('test_detail.py')
test_summary = pd.read_csv('test_summary.py')
train_detail = pd.read_csv('train_detail.py')
train_summary = pd.read_csv('train_detail.py')
customer_data = pd.read_csv('customer_data .csv')


train = pd.merge_asof(train_detail,train_summary,on = 'customer_id', by =['year','month','end_date'])
train = pd.merge_asof(train,customer_data,on='customer_id')
test = pd.merge_asof(test_detail,test_summary,on = 'customer_id', by =['year','month','end_date'])
test = pd.merge_asof(test,customer_data,on='customer_id')

train.to_csv('train.csv')
test.to_csv('test.csv')
