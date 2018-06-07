import tensorflow as tf
import pandas as pd
import math

from functions import data_processing as dp
from functions.training import train_nn_regression_model, predict
from sklearn import metrics
from functions.normalization import normalize_linear_scale, normalize_california_data

tf.logging.set_verbosity(tf.logging.FATAL)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = dp.load_data_frame_from_csv('../../data/california_housing_train.csv')

#Note that if you normalize the target, you'll need to un-normalize the predictions for loss metrics to be comparable.
normalized_dataframe = normalize_california_data(dp.preprocess_features(california_housing_dataframe))

norm_training_examples, norm_training_targets, norm_validation_examples, norm_validation_targets = dp.test_and_validation(
    california_housing_dataframe)

training_examples, training_targets, validation_examples, validation_targets = dp.test_and_validation(
    california_housing_dataframe)

# As a rule of thumb, NN's train best when the input features are roughly on the same scale.
print 'GradientDescentOptimizer'
_ = train_nn_regression_model(
    learning_rate=0,  # Actually not needed because optimizer is provided TODO refactor
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0005),
    steps=2000,
    batch_size=50,
    hidden_units=[10, 10],
    training_examples=norm_training_examples,
    training_targets=training_targets,
    validation_examples=norm_validation_examples,
    validation_targets=validation_targets)

print 'AdagradOptimizer'
_ = train_nn_regression_model(
    learning_rate=0,  # Actually not needed because optimizer is provided TODO refactor
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=norm_training_examples,
    training_targets=training_targets,
    validation_examples=norm_validation_examples,
    validation_targets=validation_targets)

print 'AdamOptimizer'
_ = train_nn_regression_model(
    learning_rate=0,  # Actually not needed because optimizer is provided TODO refactor
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
    steps=500,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=norm_training_examples,
    training_targets=training_targets,
    validation_examples=norm_validation_examples,
    validation_targets=validation_targets)

# Just linear scale
# GradientDescentOptimizer -> 129.63
# AdagradOptimizer -> 108.40
# AdamOptimizer -> 109.86

# normalize_california_data
# GradientDescentOptimizer -> 113.73
# AdagradOptimizer -> 107.57
# AdamOptimizer -> 110.15