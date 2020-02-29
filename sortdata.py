import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

customer_data = pd.read_csv("customer_data.csv").sort_values('customer_id')
sales_detail = pd.read_csv("sales_detal.csv").sort_values(['customer_id','year','month'],ascending=True)
sales_detail = pd.get_dummies(sales_detail,columns=['catagory_code'],prefix='',prefix_sep='')
sales_summary= pd.read_csv("sales_summary.csv").sort_values(['customer_id','year','month'],ascending=True)

sales = pd.merge_asof(sales_detail,sales_summary, on='customer_id', by =['year','month'])

print(customer_data.head())
print(sales_detail.head())
print(sales_summary.head())
print(sales.head())

sales.to_csv('a.csv')
