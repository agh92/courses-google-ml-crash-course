import tensorflow as tf
import pandas as pd
import math

from functions import data_processing as dp
from functions.training import train_nn_regression_model, predict
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.FATAL)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = dp.load_data_frame_from_csv('../../data/california_housing_train.csv')

training_examples, training_targets, validation_examples, validation_targets = dp.test_and_validation(
    california_housing_dataframe)

# define 2 layers with 10 and 2 nodes respectively
# ReLu is de std activation function and all nodes are fully connected
# hidden_units = [10, 2] # + learning_rate=0.01 -> 217
# hidden_units = [8, 6, 4] # + learning_rate=0.01 -> 216
# hidden_units = [8, 6, 4] # + learning_rate=0.001 -> 171
# hidden_units = [6, 4, 2] # + learning_rate=0.008 + 700 steps -> 159
hidden_units = [10, 10]  # + learning_rate=0.001 + 2000 steps -> first run 136, second run 104 curs solution

dnn = train_nn_regression_model(
    learning_rate=0.001,
    steps=2000,
    batch_size=100,
    hidden_units=hidden_units,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets,
    show=True)

california_housing_test_dataframe = dp.load_data_frame_from_csv("../../data/california_housing_test.csv")
test_set = dp.preprocess_features(california_housing_test_dataframe)
test_targets = dp.preprocess_continuous_target(california_housing_test_dataframe)
# Feed test data
predict_test_input_fn = lambda: dp.my_input_fn(
    test_set,
    test_targets['median_house_value'],
    num_epochs=1,
    shuffle=False
)

test_predictions = predict(dnn, predict_test_input_fn)
test_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
print "Final RMSE (on test data): %0.2f" % (test_root_mean_squared_error)
