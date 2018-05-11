import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("data/california_housing_train.csv", sep=",")
# randomize to prevent ordering effects
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
# scale median house value to make the learning process easier
california_housing_dataframe['median_house_value'] /= 1000
print california_housing_dataframe.describe()

###################
# BUILD THE MODEL #
###################
# input feature
total_rooms_feature = california_housing_dataframe[['total_rooms']]
# numeric feature column
feature_column_total_rooms = [tf.feature_column.numeric_column('total_rooms')]
# target we want to predict
targets = california_housing_dataframe['median_house_value']

#######################
# CONFIGURE THE MODEL #
#######################
# Define how to reduce the loss
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
# Actual prediction model
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_column_total_rooms, optimizer=my_optimizer)


# This function tells TensorFlow how to preprocess the data, as well as how to batch, shuffle, and
# repeat it during model training.
def my_input_fn(features, targets, batchsize=1, shuffle=True, num_epochs=None):
    """
    :param features: pandas DataFrame of features
    :param targets: pandas DataFrame of targets
    :param batchsize: Size of batches to be passed to the model
    :param shuffle: True or False. Whether to shuffle the data.
    :param num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    :return: Tuple of (features, labels) for next data batch
    """
    # print 'my_input_fn'
    # convert panda data into a dict of numpy arrays
    features = {key: np.array(value) for key, value in dict(features).items()}
    # Features is a dict of one key and one value total_rooms: np.array(vals)

    # Define the data set to train the model with
    data_set = Dataset.from_tensor_slices((features, targets))  # WARNING! 2GB MAX
    data_set = data_set.batch(batchsize).repeat(num_epochs)

    if shuffle:
        data_set.shuffle(buffer_size=10000)

    # define and return the batch of data
    features, labels = data_set.make_one_shot_iterator().get_next()
    return features, labels

###################
# DO THE TRAINING #
###################
# my_input_fn gets called 1 time
_ = linear_regressor.train(input_fn= lambda : my_input_fn(total_rooms_feature, targets),
                           steps=100)

############
# EVALUATE #
############
prediction_input_fn = lambda: my_input_fn(total_rooms_feature, targets, num_epochs=1, shuffle=False)
# prediction_input_fn get called 1 time
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# predictions is a list of dicts containing a key 'predictions' and a list of 1 prediction
# print predictions.next() # -> {'predictions': array([0.05919999], dtype=float32)}
predictions = np.array([item['predictions'][0] for item in predictions])

# Calculate error / Loss
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
min_house_value = california_housing_dataframe['median_house_value'].min()
max_house_value = california_housing_dataframe['median_house_value'].max()
min_max_diff = max_house_value - min_house_value

print "Min. Median House Value: %0.3f" % min_house_value
print "Max. Median House Value: %0.3f" % max_house_value
print "Difference between Min. and Max.: %0.3f" % min_max_diff
print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
print "Root Mean Squared Error: %0.3f" % root_mean_squared_error

calibration_data = pd.DataFrame()
calibration_data['predictions'] = pd.Series(predictions)
calibration_data['targets'] = pd.Series(targets)
print calibration_data.describe()

# PLOT
sample = california_housing_dataframe.sample(n=300)
# Get the min and max total_rooms values.
x_min = sample["total_rooms"].min()
x_max = sample["total_rooms"].max()
# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
# Get the predicted median_house_values for the min and max total_rooms values.
y_min = bias + weight * x_min # -> Linear function y = b + wx
y_max = bias + weight * x_max
# Plot our regression line from (x_min, y_min) to (x_max, y_max).
plt.plot([x_min, x_max], [y_min, y_max], c='r')
# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")
# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])
# Display graph.
plt.show()