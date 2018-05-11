import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# print pd.__version__

########################
# CREATE DATA MANUALLY #
########################
# Create series object -> Single column of a Dataframe
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
# The dict maps column names to respective series -> Missing values fill with NA/NaN
cities = pd.DataFrame({'City Name': city_names, 'Population':population})

#################################
# ACCES DATA WITH dict/list OPS #
#################################
# print type(cities['Population'])
# print type(cities['Population'][1])  # Extensive indexing and accesing posiblilites http://pandas.pydata.org/pandas-docs/stable/indexing.html

#########################
# ARITHNETIC OPERATIONS #
#########################
# for p in population:
#     print p
# population /= 1000
# for p in population:
#     print p
# population = np.log(population)
# for p in population:
#     print p
# population_over_mlln = population.apply(lambda x: x if x > 1000000 else None)
# for p in population_over_mlln:
#      print p

#####################
# MODIFY DATAFRAMES #
#####################
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
# print cities

###########
# INDEXES #
###########
# print cities.index
# cities = cities.reindex(np.random.permutation(cities.index))

##############
# EXERCISE 1 #
##############
cities['Named after Saint and is big'] = (cities['Area square miles'] > 50) & cities['City Name'].apply(lambda name: name.find('San ') != -1)

##############
# EXERCISE 2 #
##############
# cities = cities.reindex([2, 1, 0, 6])  # missing values are filled with NaN
cities = cities.reindex([2, 1, 0, 6], fill_value='Missing')
print cities


#########################
# LOAD DATA FROM SOURCE #
#########################
california_housing_dataframe = pd.read_csv("../data/california_housing_train.csv", sep=",")
# Statistics
# print california_housing_dataframe.describe()
# First records
# print california_housing_dataframe.head()
# Explore distribution of a certain column -> should plot a graph
# california_housing_dataframe.hist('housing_median_age')
# Tell matplotlib to show the graph
# plt.show()