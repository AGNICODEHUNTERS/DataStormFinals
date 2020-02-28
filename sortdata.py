import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

customer_data = pd.read_csv("customer_data.csv").sort_values('customer_id')
sales_detail = pd.read_csv("sales_detal.csv").sort_values(['customer_id','year','month'],ascending=True)
sales_summary= pd.read_csv("sales_summary.csv").sort_values(['customer_id','year','month'],ascending=True)

print(customer_data.head())
print(sales_detail.head())
print(sales_summary.head())
