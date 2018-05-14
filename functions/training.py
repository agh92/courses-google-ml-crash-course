import math
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import functions.data_processing as dp
from functions import ploting


def custom_linear_regressor(learning_rate, feature_columns):
    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    return tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )


# convenience function
def predict(linear_regressor, input_fn):
    training_predictions = linear_regressor.predict(input_fn=input_fn)
    return np.array([item['predictions'][0] for item in training_predictions])


def train_model_single_feature(
        data_frame,
        learning_rate,
        steps,
        batch_size,
        input_feature="total_rooms",
        my_target="median_house_value",
        show=False):
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
    my_feature_data = data_frame[[input_feature]].astype('float32')
    targets = data_frame[my_target].astype('float32')

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(input_feature)]

    # Create input functions.
    training_input_fn = lambda: dp.my_input_fn(my_feature_data, targets, batchsize=batch_size)
    prediction_input_fn = lambda: dp.my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    linear_regressor = custom_linear_regressor(learning_rate, feature_columns)

    if show:
        # Set up to plot the state of our model's line each period.
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sample = data_frame.sample(n=300)
        colors = ploting.plot_sample(sample=sample, my_feature=input_feature,
                                     my_label=my_target, periods=periods)

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
        predictions = predict(linear_regressor, prediction_input_fn)

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, root_mean_squared_error)
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)

        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        if show:
            ploting.plot_linear_model(sample, my_target, linear_regressor, input_feature, colors[period])
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    if show:
        plt.subplot(1, 2, 2)
        ploting.plot_loss_over_periods({'training': root_mean_squared_errors})
        plt.show()

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print "Final RMSE (on training data): %0.2f" % root_mean_squared_error
    return calibration_data


def train_model_multi_feature(
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        learning_rate,
        steps,
        batch_size,
        show=False):
    periods = 10
    steps_per_period = steps / periods

    linear_regressor = custom_linear_regressor(learning_rate, dp.construct_feature_columns(training_examples))

    # Feed for training
    training_input_fn = lambda: dp.my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        batchsize=batch_size
    )
    # Feed to make predictions
    predict_training_input_fn = lambda: dp.my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        num_epochs=1,
        shuffle=False
    )
    # Feed to make validation
    predict_validation_input_fn = lambda: dp.my_input_fn(
        validation_examples,
        validation_targets['median_house_value'],
        num_epochs=1,
        shuffle=False
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "RMSE (on training data):"
    validation_rmse, training_rmse = [], []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # 2. Take a break and compute predictions.
        training_predictions = predict(linear_regressor, predict_training_input_fn)
        validation_predictions = predict(linear_regressor, predict_validation_input_fn)

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, training_root_mean_squared_error)
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    if show:
        ploting.plot_loss_over_periods({"training": training_rmse, "validation": validation_rmse})
        plt.show()

    return linear_regressor
