from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from functions.training import train_model_multi_feature
from functions.data_processing import load_data_frame_from_csv

#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = load_data_frame_from_csv("../../data/california_housing_train.csv")

# scale median house value to make the learning process easier
california_housing_dataframe['median_house_value'] /= 1000
# print california_housing_dataframe.describe()

##########
# TASK 1 #
##########
california_housing_dataframe["rooms_per_person"] = (
        california_housing_dataframe['total_rooms'] / california_housing_dataframe['population'])
# print california_housing_dataframe.describe()
calibration_data = train_model_multi_feature(
    data_frame=california_housing_dataframe,
    learning_rate=0.04,
    steps=500,
    batch_size=5,
    input_features=["rooms_per_person"],
    show=False
)

##########
# TASK 2 #
##########
# scatter plot of predictions vs. target values
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
# this diagram shows that mos of the points are align in a vertical line and there are some outliers
plt.scatter(calibration_data["predictions"], calibration_data["targets"])
# we confirm the diagram above seeing the distribution of the rooms per person histogram
plt.subplot(1, 2, 2)
_ = california_housing_dataframe["rooms_per_person"].hist()
# plt.show()

##########
# TASK 3 #
##########

# From california_housing_dataframe.describe() after adding the syntetic feature rooms_per_person
#        rooms_per_person
# count           17000.0
# mean                2.0
# std                 1.2
# min                 0.0
# 25%                 1.5
# 50%                 1.9
# 75%                 2.3
# max                55.2
# taking the values into account -> Choose 5 as a limit for outliers (coincidental the same as the course)
clipped_rooms_per_person = california_housing_dataframe["rooms_per_person"].apply(lambda x: min(x, 5))
# could also have replaced rooms_per_person with the clipped data as in the course solution
california_housing_dataframe["clipped_rooms_per_person"] = clipped_rooms_per_person
# Final RMSE = 111.83
calibration_data_2 = train_model_multi_feature(
    data_frame=california_housing_dataframe,
    learning_rate=0.04,
    steps=500,
    batch_size=5,
    input_features=["clipped_rooms_per_person"],
    show=False
)
_ = california_housing_dataframe["clipped_rooms_per_person"].hist()

plt.figure(figsize=(15, 6))
_ = plt.scatter(calibration_data_2['predictions'], calibration_data_2['targets'])

plt.show()