import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset


def load_data_frame_from_csv(location, sep=","):
    """

    :param location:
    :param sep:
    :return:
    """
    df = pd.read_csv(location, sep=sep)
    return df.reindex(np.random.permutation(df.index))


def construct_feature_columns(input_features):
    """
    :param input_features: Names of numerical input features to use
    :return: set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])


def my_input_fn(features, targets, batchsize=1, shuffle=True, num_epochs=None):
    """This function tells TensorFlow how to preprocess the data, as well as how to batch, shuffle, and
        repeat it during model training.

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


def test_and_validation(data_frame, examples_percentage=0.7, binary=False):
    """

    :param data_frame:
    :param examples_percentage:
    :return:
    """
    count = len(data_frame.index)
    examples_len = math.trunc(count * examples_percentage)
    validation_len = math.trunc(count * (1.0 - examples_percentage))

    training_examples = preprocess_features(data_frame.head(examples_len))
    validation_examples = preprocess_features(data_frame.tail(validation_len))

    if not binary:
        training_targets = preprocess_continuous_target(data_frame.head(examples_len))
        validation_targets = preprocess_continuous_target(data_frame.tail(validation_len))
    else:
        training_targets = preprocess_binary_target(data_frame.head(examples_len))
        validation_targets = preprocess_binary_target(data_frame.tail(validation_len))

    return training_examples, training_targets, validation_examples, validation_targets


# Following functions are only for the examples of california housing data
def preprocess_features(data_frame):
    """
    :param data_frame: Data loaded from california housing examples as a DataFrame
    :return: DataFrame containing selected and synthetic features of the california housing Data
    """
    selected_features = data_frame[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]]

    processed_features = selected_features.copy()
    # synthetic feature
    processed_features["rooms_per_person"] = (
            data_frame["total_rooms"] /
            data_frame["population"])

    return processed_features


def preprocess_binary_target(data_frame):
    output_targets = pd.DataFrame()
    output_targets['median_house_value_is_high'] = (data_frame['median_house_value'] > 265000).astype(float)
    return output_targets


def preprocess_continuous_target(data_frame):
    """
    :param data_frame: Data loaded from california housing examples as a DataFrame
    :return: DataFrame containing target to predict
    """
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = (
            data_frame['median_house_value'] / 1000.0)
    return output_targets
