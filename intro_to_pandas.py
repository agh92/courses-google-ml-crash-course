import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print pd.__version__

# MANUAL VERSION OF LOADING DATA
# Create series object -> Single column of a Dataframe
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
# The dict maps column names to respective series -> Missing values fill with NA/NaN
cities = pd.DataFrame({'City Name': city_names, 'Population':population})
# Acces data with standard dict/list operations
print type(cities['Population'][1])  # Extensive indexing and accesing posiblilites http://pandas.pydata.org/pandas-docs/stable/indexing.html


# Load data from csv
california_housing_dataframe = pd.read_csv("data/california_housing_train.csv", sep=",")
# Statistics
# print california_housing_dataframe.describe()
# First records
# print california_housing_dataframe.head()
# Explore distribution of a certain column -> should plot a graph
# california_housing_dataframe.hist('housing_median_age')
# Tell matplotlib to show the graph
# plt.show()