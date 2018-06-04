import tensorflow as tf
import pandas as pd
from functions import data_processing as dp

tf.logging.set_verbosity(tf.logging.FATAL)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = dp.load_data_frame_from_csv('../data/california_housing_train.csv')

training_examples, training_targets, validation_examples, validation_targets = dp.test_and_validation(
    california_housing_dataframe)

# define 2 layers with 3 and 10 nodes respectively
# ReLu is de std activation function and all nodes are fully connected
hidden_units = [3, 10]
