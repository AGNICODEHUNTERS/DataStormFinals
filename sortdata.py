import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

customer_data = pd.read_csv("customer_data.csv").sort_values('customer_id').dropna()
sales_detail = pd.read_csv("sales_detal.csv").sort_values(['customer_id','year','month'],ascending=True)
sales_detail = pd.get_dummies(sales_detail,columns=['catagory_code'],prefix='',prefix_sep='')
sales_summary= pd.read_csv("sales_summary.csv").sort_values(['customer_id','year','month'],ascending=True)


sales_summary = sales_summary.groupby('customer_id')
sales = sales_summary.get_group(167939)
'''sales = pd.merge_asof(right = sales_detail,left = sales_summary, on='customer_id', by =['year','month','end_date'])
sales = pd.merge_asof(sales,customer_data,on='customer_id')
print(customer_data.head())
print(sales_detail.head())
print(sales_summary.head())
print(sales.head())'''

sales.to_csv('a.csv',index=False)
