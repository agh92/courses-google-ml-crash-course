import tensorflow as tf
import pandas as pd
from functions import data_processing as dp
from functions import training as tr
from functions import normalization as cl

#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load the train and validation data
california_housing_dataframe = dp.load_data_frame_from_csv("../data/california_housing_train.csv")
training_examples, training_targets, validation_examples, validation_targets = dp.test_and_validation(
    california_housing_dataframe)

# print training_examples.describe()
# print validation_examples.describe()
# print validation_set.describe()

# _ = tr.train_model_all_features(
#     learning_rate=1.0,
#     steps=500,
#     batch_size=100,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets,
#     optimizer=tf.train.FtrlOptimizer,
#     show=True)
#
# FINAL RMSE (without feature cross) -> 65.86
# FINAL RMSE (with feature cross) -> 61.85
_ = tr.train_model_all_features(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=cl.bucketize_feature_columns(training_examples,
                                                 ["households", "longitude", "latitude", "housing_median_age",
                                                  "median_income", "rooms_per_person"], [['latitude', 'longitude']]),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
