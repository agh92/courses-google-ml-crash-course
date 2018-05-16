import math

import pandas as pd
import tensorflow as tf
import functions.data_processing as dp
from functions import training as tr
from matplotlib import pyplot as plt
from functions import cleaning as cl

#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load the train and validation data
california_housing_dataframe = dp.load_data_frame_from_csv("../data/california_housing_train.csv")

training_examples = dp.preprocess_features(california_housing_dataframe.head(12000))
training_targets = dp.preprocess_targets(california_housing_dataframe.head(12000))

training_examples['long_over_lat'] = training_examples['longitude'] / training_examples['latitude']
training_examples['income_over_lat'] = training_examples['median_income'] / training_examples['latitude']

# print training_examples.describe()
# print training_targets.describe()

validation_examples = dp.preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = dp.preprocess_targets(california_housing_dataframe.tail(5000))

validation_examples['long_over_lat'] = validation_examples['longitude'] / validation_examples['latitude']
validation_examples['income_over_lat'] = validation_examples['median_income'] / validation_examples['latitude']
# print validation_examples.describe()
# print validation_set.describe()

correlation_dataframe = training_examples.copy()
correlation_dataframe['target'] = training_targets['median_house_value']
# -1: perfect negative correlation
# 0: no correlation
# 1: perfect positive correlation
print correlation_dataframe.corr()

plt.scatter(training_examples['latitude'], training_targets['median_house_value'])
plt.show()


minimal_features = [
    'median_income',
    # 'housing_median_age',
    # 'rooms_per_person',
    'latitude'
    # 'long_over_lat',
    # 'income_over_lat'
]

assert minimal_features, "You must select at least one feature!"

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

# Define the range of the binning
min_latitude = math.trunc(minimal_training_examples['latitude'].min())
max_latitude = int(math.ceil(minimal_training_examples['latitude'].max()))

# LATITUDE_RANGES = zip(xrange(32, 44), xrange(33, 45))
LATITUDE_RANGES = zip(xrange(min_latitude, max_latitude), xrange(min_latitude + 1, max_latitude + 1))

minimal_training_examples = cl.binning_feature(minimal_training_examples, LATITUDE_RANGES, 'latitude')
minimal_validation_examples = cl.binning_feature(minimal_validation_examples, LATITUDE_RANGES, 'latitude')

print minimal_training_examples.describe()
print minimal_validation_examples.describe()

#
# Don't forget to adjust these parameters.
# learning_rate=0.001 steps=500 batch_size=5
# 'median_income' 'housing_median_age' 'rooms_per_person' -> 179.27
#
# learning_rate=0.001 steps=700 batch_size=5
# 'median_income' 'housing_median_age' 'rooms_per_person' -> 158.40
#
# Solution of course
# learning_rate=0.01 steps=500 batch_size=5
# 'median_income' 'latitude' -> 113.16
#
# Solution of course
# learning_rate=0.01 steps=500 batch_size=5
# 'income_over_lat' 'long_over_lat' -> 166.34
#
# binning_feature -> 139.81
# binning_feature using dynamic range -> 111.36
#
tr.train_model_multi_feature(
    learning_rate=0.01,
    steps=500,
    batch_size=5,
    training_examples=minimal_training_examples,
    training_targets=training_targets,
    validation_examples=minimal_validation_examples,
    validation_targets=validation_targets)