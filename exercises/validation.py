import math

import numpy as np
import pandas as pd
import tensorflow as tf
import functions.data_processing as dp
from functions import training as tr
from sklearn import metrics


#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

##########
# TASK 3 #
##########
# Load the train and validation data
california_housing_dataframe = dp.load_data_frame_from_csv("../data/california_housing_train.csv")

training_set = dp.preprocess_features(california_housing_dataframe.head(12000))
print training_set.describe()
training_targets = dp.preprocess_targets(california_housing_dataframe.head(12000))
#print training_targets.describe()

validation_set = dp.preprocess_features(california_housing_dataframe.tail(5000))
print validation_set.describe()
validation_targets = dp.preprocess_targets(california_housing_dataframe.tail(5000))
#print validation_set.describe()

##########
# TASK 1 #
##########
# Does the data fulfills the expectations?
# Are there odd things?

##########
# TASK 2 #
##########
# The key thing to notice is that for any given feature or column, the distribution of values between the train and
# validation splits should be roughly equal !!! -> Fix in Task 3
# plt.figure(figsize=(13, 8))
#
# ax = plt.subplot(1, 2, 1)
# ax.set_title("Validation Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(validation_set["longitude"],
#             validation_set["latitude"],
#             cmap="coolwarm",
#             c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
#
# ax = plt.subplot(1,2,2)
# ax.set_title("Training Data")
#
# ax.set_autoscaley_on(False)
# ax.set_ylim([32, 43])
# ax.set_autoscalex_on(False)
# ax.set_xlim([-126, -112])
# plt.scatter(training_set["longitude"],
#             training_set["latitude"],
#             cmap="coolwarm",
#             c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
# _ = plt.plot()
# plt.show()

##########
# TASK 4 #
##########
# Train the modell
linear_regressor = tr.train_model_all_feature(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_set,
    training_targets=training_targets,
    validation_examples=validation_set,
    validation_targets=validation_targets)


##########
# TASK 5 #
##########
# Load test data
california_housing_test_dataframe = dp.load_data_frame_from_csv("../../data/california_housing_test.csv")
test_set = dp.preprocess_features(california_housing_test_dataframe)
test_targets = dp.preprocess_targets(california_housing_test_dataframe)
# Feed test data
predict_test_input_fn = lambda: dp.my_input_fn(
    test_set,
    test_targets['median_house_value'],
    num_epochs=1,
    shuffle=False
)
# make predictions with test data
test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])
# check the error with the test data
test_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
print "Final RMSE (on test data): %0.2f" % (test_root_mean_squared_error)