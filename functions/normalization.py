import math
import numpy as np
import tensorflow as tf
import pandas as pd


# WARNING only compatible with california housing data
def normalize_california_data(examples_dataframe):
    processed_features = pd.DataFrame()
    # households, median_income and total_bedrooms all appear normally-distributed in a log space
    processed_features["households"] = log_normalize(examples_dataframe["households"])
    processed_features["median_income"] = log_normalize(examples_dataframe["median_income"])
    processed_features["total_bedrooms"] = log_normalize(examples_dataframe["total_bedrooms"])
    # latitude, longitude and housing_median_age would probably be better off just scaled linearly
    processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    # population, totalRooms and rooms_per_person have a few extreme outliers. They seem too extreme for log
    # normalization to help. So let's clip them instead.
    processed_features["population"] = linear_scale(clip(examples_dataframe["population"], 0, 5000))
    processed_features["rooms_per_person"] = linear_scale(clip(examples_dataframe["rooms_per_person"], 0, 5))
    processed_features["total_rooms"] = linear_scale(clip(examples_dataframe["total_rooms"], 0, 10000))

    return processed_features


def log_normalize(series):
    return series.apply(lambda x: math.log(x + 1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: (min(max(x, clip_to_min), clip_to_max)))


def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x - mean) / std_dv)


def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))


def normalize_linear_scale(examples_dataframe):
    df = pd.DataFrame()
    for series in examples_dataframe:
        df[series] = linear_scale(examples_dataframe[series])
    return df


def linear_scale(series):
    """
    It can be a good standard practice to normalize the inputs to fall within the range -1, 1. This helps SGD not get
    stuck taking steps that are too large in one dimension, or too small in another.
    :param series:
    :return:
    """
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


def binning_feature(source_df, feature, ranges=None):
    """ Divide a feature into "bins" according to the ranges

    :param source_df: DataFrame containing the feature to bin
    :param ranges: ranges for the binning. If not provided min and max values from the selected feature
        will be used to define de range.
    :param feature: Feature to be divided in "bins"
    :return: DataFrame containing the "bins" as extra features
    """

    if ranges is None:
        # Define the range of the binning
        min = math.trunc(source_df[feature].min())
        max = int(math.ceil(source_df[feature].max()))

        # LATITUDE_RANGES = zip(xrange(32, 44), xrange(33, 45))
        ranges = zip(xrange(min, max), xrange(min + 1, max + 1))

    selected_examples = source_df.copy()
    for r in ranges:
        selected_examples[(feature + "_%d_to_%d") % r] = source_df[feature].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples


def get_quantile_based_boundaries(feature_values, num_buckets):
    """

    :param feature_values:
    :param num_buckets:
    :return:
    """
    boundaries = np.arange(start=1.0, stop=num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]


feature_bucktes_count = {
    "households": 10,
    "longitude": 50,
    "latitude": 50,
    "housing_median_age": 10,
    "median_income": 10,
    "total_rooms": 10,
    "total_bedrooms": 10,
    "rooms_per_person": 10,
    "population": 10
}


def bucketize_feature_columns(training_examples, features, cross_features=None):
    """Construct the TensorFlow Feature Columns.
    :param cross_features: list of lists containing features to cross together
    Returns:
      A set of feature columns

    """

    if cross_features is None:
        cross_features = []

    feature_columns = set()
    # TASK 1 OF FEATURE_CROSS THE COMMENTED LINE SHOULD BE A BETTER AND DYNAMIC WAY TO GET THE NUMBER OF BUCKETS
    # In the exercises of the curs they use different number of buckets for different features,
    # to save some code all feature will have the same number of buckets
    buckets_count = 10  # math.ceil(math.sqrt(len(training_examples.index))) -> Should give the best number of buckets but does not work

    feature_bucket = dict()

    for feature in features:
        numeric_feature = tf.feature_column.numeric_column(feature)
        bucketized_feature = tf.feature_column.bucketized_column(
            numeric_feature,
            boundaries=get_quantile_based_boundaries(training_examples[feature], feature_bucktes_count[feature]))
        feature_bucket[feature] = bucketized_feature
        feature_columns.add(bucketized_feature)

    for features_list in cross_features:
        buckets = []
        for feature in features_list:
            buckets.append(feature_bucket[feature])
        feature_columns.add(tf.feature_column.crossed_column(buckets, 1000))

    return feature_columns
