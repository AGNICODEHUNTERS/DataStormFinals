import pandas as pd

sales_detail = pd.read_csv("sales_detail .csv").sort_values(['customer_id','year','month'],ascending=True)
sales_detail = pd.get_dummies(sales_detail,columns=['catagory_code'],prefix='',prefix_sep='')

sales_detail.to_csv('full_sales_detail.csv',index=False)
