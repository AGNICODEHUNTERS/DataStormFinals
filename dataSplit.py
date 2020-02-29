import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

sales_detail = pd.read_csv("full_sales_detail.csv").sort_values(['customer_id','year','month'],ascending=True)
sales_summary= pd.read_csv("sales_summary .csv").sort_values(['customer_id','year','month'],ascending=True)
sales_detail = sales_detail.groupby(['customer_id','year','month']).sum()

sales_summary_ed = sales_summary.groupby(['year','month'])
sales_summary_test = sales_summary_ed.get_group((2019,12))
sales_summary_train = sales_summary.drop(index = sales_summary_test.index)

sales_detail_ed = sales_detail.groupby(['year','month'])
sales_detail_test = sales_detail_ed.get_group((2019,12))
print(sales_detail_test)
sales_detail_train = sales_detail.drop(index = sales_detail_test.index)

sales_summary_test.to_csv('test_summary.csv',index=False)
sales_summary_train.to_csv('train_summary.csv',index = False)

sales_detail_test.to_csv('test_detail.csv')
sales_detail_train.to_csv('train_detail.csv')
