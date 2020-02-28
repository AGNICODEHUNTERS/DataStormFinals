import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

customer_data = pd.read_csv("customer_data.csv")
sales_detail = pd.read_csv("sales_detal.csv")
sales_summary= pd.read_csv("sales_summary.csv")

print(customer_data.head())
print(sales_detail.head())
print(sales_summary.head())
