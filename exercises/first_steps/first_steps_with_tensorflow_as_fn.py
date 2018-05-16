
import pandas as pd
import tensorflow as tf
from functions.training import train_model_single_feature
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


# this combination gets a RMSE of 225.63
# train_model_single_feature(
#     data_frame=california_housing_dataframe,
#     learning_rate=0.00001,
#     steps=100,
#     batch_size=1,
#     input_feature=["total_rooms"]
# )

#####################################
# TASK 1 -> GET RMSE LOWER THAN 180 #
#####################################
# Final RMSE = 166.74
# train_model_single_feature(
#     data_frame=california_housing_dataframe,
#     learning_rate=0.0001,
#     steps=100,
#     batch_size=1,
#     input_feature=["total_rooms"]
# )

# Solution of the course -> RMSE = 167.79 -> Higher tha  above
# train_model_single_feature(
#     data_frame=california_housing_dataframe,
#     learning_rate=0.00002,
#     steps=500,
#     batch_size=1,
#     input_feature=["total_rooms"]
# )

##################################################################################################
# TASK 2                                                                                         #
# See if you can do any better by replacing the total_rooms feature with the population feature. #
##################################################################################################
# Final RMSE -> 176.84
train_model_single_feature(
    data_frame=california_housing_dataframe,
    learning_rate=0.00003,
    steps=900,
    input_features=['population'],
    batch_size=4
)

# COURSE SOLUTION
# Final RMSE -> 175.97
# train_model_single_feature(
#     data_frame=california_housing_dataframe,
#     learning_rate=0.00002,
#     steps=1000,
#     batch_size=5,
#     input_feature=["population"]
# )
