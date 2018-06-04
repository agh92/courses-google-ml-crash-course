import math
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import functions.data_processing as dp
from functions import ploting


def predict(linear_regressor, input_fn):
    """Convenience function

    :param linear_regressor:
    :param input_fn:
    :return:
    """
    training_predictions = linear_regressor.predict(input_fn=input_fn)
    return np.array([item['predictions'][0] for item in training_predictions])


def train_model_multi_feature(
        data_frame,
        learning_rate,
        steps,
        batch_size,
        input_features,
        my_target="median_house_value",
        show=False):
    """Trains a linear regression model of one feature.

        total_number_of_trained_examples = steps * batch_size

    :param data_frame: A `DataFrame` object to work with.
    :param learning_rate: A `float`, the learning rate.
    :param steps: A non-zero `int`, the total number of training steps/iterations. A training step
        consists of a forward and backward pass using a single batch.
    :param batch_size: A non-zero `int`, the batch size. Numbers of examples for a single step
    :param input_features: A `string` or `list` of strings specifying the columns to use as input feature.
    :param my_target: A `string` specifying a column to use as target.
    :param show: A `string` specifying whether to plot or not.
    :return: A `DataFrame`containing the predictions and the targets to use as calibration data.
    """

    my_feature_data = data_frame[input_features]
    targets = data_frame[[my_target]]

    # As the data was not separated into validation and example sets do it
    count = len(data_frame.index)
    examples_len = math.trunc(count * 0.7)
    validation_len = math.trunc(count * 0.3)
    # dont use dp.test_and_validation() because it would use all features
    training_examples = my_feature_data.head(examples_len)
    training_targets = dp.preprocess_continuous_target(targets.head(examples_len))
    validation_examples = my_feature_data.tail(validation_len)
    validation_targets = dp.preprocess_continuous_target(targets.tail(validation_len))

    linear_regressor = train_model_all_features(
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets,
        learning_rate=learning_rate,
        steps=steps,
        batch_size=batch_size,
        show=show)

    predictions = predict(linear_regressor,
                          lambda: dp.my_input_fn(training_examples, training_targets, num_epochs=1, shuffle=False))

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(training_targets)
    display.display(calibration_data.describe())

    # print "Final RMSE (on training data): %0.2f" % root_mean_squared_error
    return calibration_data


def train_model_all_features(
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        learning_rate,
        steps,
        batch_size,
        feature_columns=None,
        optimizer=tf.train.GradientDescentOptimizer,
        show=False,
        my_target="median_house_value"):
    """Trains a linear regression model using several features.

    :param feature_columns:
    :param optimizer:
    :param my_target: A `string` specifying a column to use as target.
    :param training_examples:
    :param training_targets:
    :param validation_examples:
    :param validation_targets:
    :param learning_rate: A `float`, the learning rate.
    :param steps: A non-zero `int`, the total number of training steps/iterations. A training step
        consists of a forward and backward pass using a single batch.
    :param batch_size: A non-zero `int`, the batch size. Numbers of examples for a single step
    :param show: A `string` specifying whether to plot or not.
    :return: The trained linear regression model
    """
    # periods controls the granularity of reporting -> modifying periods does not alter what your model learns
    # trained_examples_per_period = (batch_size * steps) / periods
    periods = 10
    # the exercise will output the loss value every (steps / periods) steps
    steps_per_period = steps / periods

    my_optimizer = optimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=dp.construct_feature_columns(training_examples) if feature_columns is None else feature_columns,
        optimizer=my_optimizer
    )

    # Feed for training
    training_input_fn = lambda: dp.my_input_fn(
        training_examples,
        training_targets[my_target],
        batchsize=batch_size
    )
    # Feed to make predictions
    predict_training_input_fn = lambda: dp.my_input_fn(
        training_examples,
        training_targets[my_target],
        num_epochs=1,
        shuffle=False
    )
    # Feed to make validation
    predict_validation_input_fn = lambda: dp.my_input_fn(
        validation_examples,
        validation_targets[my_target],
        num_epochs=1,
        shuffle=False
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "RMSE:"
    validation_rmse, training_rmse, validation_loglosses, training_loglosses = [], [], [], []
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
        print "  period %02d : %0.2f RMSE" % (period, training_root_mean_squared_error)
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    if show:
        ploting.plot_loss_over_periods({"training": training_rmse,
                                        "validation": validation_rmse})
        plt.show()

    return linear_regressor


def train_linear_classifier_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,
        label="median_house_value_is_high",
        show=False):

    periods = 10
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier =  tf.estimator.LinearClassifier(
        feature_columns=dp.construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: dp.my_input_fn(training_examples,
                                               training_targets[label],
                                               batchsize=batch_size)
    predict_training_input_fn = lambda: dp.my_input_fn(training_examples,
                                                    training_targets[label],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: dp.my_input_fn(validation_examples,
                                                      validation_targets[label],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print "Training model..."
    print "LogLoss (on training data):"
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, training_log_loss)
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print "Model training finished."

    if show:
        ploting.plot_loss_over_periods({"training": training_log_losses,
                                        "validation": validation_log_losses})
        plt.show()
    return linear_classifier
