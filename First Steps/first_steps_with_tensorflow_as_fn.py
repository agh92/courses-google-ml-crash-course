import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
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
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
# scale median house value to make the learning process easier
california_housing_dataframe['median_house_value'] /= 1000
# print california_housing_dataframe.describe()


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


# This function was taken 1:1 from the course
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batchsize=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "RMSE (on training data):"
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, root_mean_squared_error)
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print "Final RMSE (on training data): %0.2f" % root_mean_squared_error


# this combination gets a RMSE of 225.63
# train_model(
#     learning_rate=0.00001,
#     steps=100,
#     batch_size=1
# )

#####################################
# TASK 1 -> GET RMSE LOWER THAN 180 #
#####################################
# Final RMSE = 166.74
# train_model(
#     learning_rate=0.0001,
#     steps=100,
#     batch_size=1
# )

# Solution of the course -> RMSE = 167.79 -> Higher tha  above
# train_model(
#     learning_rate=0.00002,
#     steps=500,
#     batch_size=1
# )

##################################################################################################
# TASK 2                                                                                         #
# See if you can do any better by replacing the total_rooms feature with the population feature. #
##################################################################################################
# Final RMSE -> 176.84
train_model(
    learning_rate=0.00003,
    steps=900,
    input_feature='population',
    batch_size=4
)

# COURSE SOLUTION
# Final RMSE -> 175.97
# train_model(
#     learning_rate=0.00002,
#     steps=1000,
#     batch_size=5,
#     input_feature="population"
# )