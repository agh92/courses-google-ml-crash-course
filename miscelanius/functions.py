import math
from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


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


def train_model(data_frame, learning_rate, steps, batch_size, input_feature="total_rooms", show=True):
    """Trains a linear regression model of one feature.

    total_number_of_trained_examples = steps * batch_size

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps/iterations. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size. Numbers of examples for a single step
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """
    # periods controls the granularity of reporting -> modifying periods does not alter what your model learns
    periods = 10
    # the exercise will output the loss value every (steps / periods) steps
    steps_per_period = steps / periods

    # trained_examples_per_period = (batch_size * steps) / periods

    my_feature = input_feature
    my_feature_data = data_frame[[my_feature]].astype('float32')
    my_label = "median_house_value"
    targets = data_frame[my_label].astype('float32')

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
    sample = data_frame.sample(n=300)
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
    if show:
        plt.show()

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print "Final RMSE (on training data): %0.2f" % root_mean_squared_error
    return calibration_data
