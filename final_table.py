import pandas as pd

test_detail = pd.read_csv('test_detail.csv').sort_values(['customer_id','year','month'],ascending=True).drop('end_date',axis=1)
test_summary = pd.read_csv('test_summary.csv').sort_values(['customer_id','year','month'],ascending=True).drop('end_date',axis=1)
train_detail = pd.read_csv('train_detail.csv').sort_values(['customer_id','year','month'],ascending=True).drop('end_date',axis=1)
train_summary = pd.read_csv('train_summary.csv').sort_values(['customer_id','year','month'],ascending=True).drop('end_date',axis=1)
customer_data = pd.read_csv('customer_data .csv').sort_values('customer_id',ascending=True)


train = pd.merge_asof(train_detail,train_summary,on = 'customer_id',by=['year','month'])
train = pd.merge_asof(train,customer_data,on='customer_id')
test = pd.merge_asof(test_detail,test_summary,on = 'customer_id',by=['year','month'])
test = pd.merge_asof(test,customer_data,on='customer_id')

train.to_csv('train.csv',index = False)
test.to_csv('test.csv',index = False)
