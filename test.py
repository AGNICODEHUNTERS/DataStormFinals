# importing pandas as pd
import pandas as pd

# Creating the Series
sr = pd.Series([1, 0, 0;0, 1, 0;0, 1, 1])

# Create the Index
index_ = ['Coca Cola', 'Sprite', 'Coke', 'Fanta', 'Dew', 'ThumbsUp']

# set the index
sr.index = index_

# Print the series
print(sr)
# multiply the given value with series
result = sr.multiply(other = [1,2,3,4,5,6])

# Print the result
print(result)
