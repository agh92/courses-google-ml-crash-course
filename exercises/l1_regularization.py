import tensorflow as tf
import pandas as pd
from functions import data_processing as dp
from functions import training as tr
from functions.normalization import bucketize_feature_columns
from functions.utils import model_size

tf.logging.set_verbosity(tf.logging.FATAL)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = dp.load_data_frame_from_csv('../data/california_housing_train.csv')

training_examples, training_targets, validation_examples, validation_targets = dp.test_and_validation(
    california_housing_dataframe, binary=True)

linear_classifier = tr.train_linear_classifier_model_l1(
    learning_rate=0.1,
    # TWEAK THE REGULARIZATION VALUE BELOW -> 0.3: 648 Dimensions, 0.15: 683 Dimensions, 0.1:684
    regularization_strength=0.3,
    steps=300,
    batch_size=100,
    feature_columns=bucketize_feature_columns(training_examples,
                                              ["households",
                                               "longitude",
                                               "latitude",
                                               "housing_median_age",
                                               "median_income",
                                               "total_rooms",
                                               "total_bedrooms",
                                               "rooms_per_person",
                                               "population"
                                               ], [["longitude", "latitude"]]),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
print "Model size:", model_size(linear_classifier)
