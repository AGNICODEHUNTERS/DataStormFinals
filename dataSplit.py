import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

customer_data = pd.read_csv("customer_data .csv").sort_values('customer_id').dropna()
sales_detail = pd.read_csv("sales_detail .csv").sort_values(['customer_id','year','month'],ascending=True)
sales_detail = pd.get_dummies(sales_detail,columns=['catagory_code'],prefix='',prefix_sep='')
sales_summary= pd.read_csv("sales_summary .csv").sort_values(['customer_id','year','month'],ascending=True)

churn_arr=[]
sales_summary_ed = sales_summary.groupby(['year','month'])
sales_summary_test = sales_summary_ed.get_group((2019,12))
sales_summary_train = sales_summary.drop(index = sales_summary_test.index)

sales_summary_test.to_csv('train_summary.csv',index=False)
sales_summary_train.to_csv('test_summary.csv',index = False)
