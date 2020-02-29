import pandas as pd

test_detail = pd.read_csv('test_detail.csv').sort_values(['customer_id','year','month'],ascending=True)
test_summary = pd.read_csv('test_summary.csv').sort_values(['customer_id','year','month'],ascending=True)
train_detail = pd.read_csv('train_detail.csv').sort_values(['customer_id','year','month'],ascending=True)
train_summary = pd.read_csv('train_summary.csv').sort_values(['customer_id','year','month'],ascending=True)
customer_data = pd.read_csv('customer_data .csv').sort_values('customer_id',ascending=True)
customer_data=pd.get_dummies(customer_data,columns=['gender'])

train = pd.merge_asof(train_detail,train_summary,on = 'customer_id',by=['year','month'])
train = pd.merge_asof(train,customer_data,on='customer_id')
train = train.drop('end_date',axis=1)
test = pd.merge_asof(test_detail,test_summary,on = 'customer_id',by=['year','month'])
test = pd.merge_asof(test,customer_data,on='customer_id')

train.to_csv('train.csv',index = False)
test.to_csv('test.csv',index = False)
