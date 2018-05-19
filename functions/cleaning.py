import math
import numpy as np
import tensorflow as tf


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


def bucketize_feature_columns(training_examples, features, cross_features=None):
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """

    feature_columns = set()
    # TASK 1 OF FEATURE_CROSS THE COMMENTED LINE SHOULD BE A BETTER AND DYNAMIC WAY TO GET THE NUMBER OF BUCKETS
    buckets_count = 10  # math.trunc(math.sqrt(len(training_examples.index)))

    for feature in features:
        numeric_feature = tf.feature_column.numeric_column(feature)
        bucketized_feature = tf.feature_column.bucketized_column(
            numeric_feature, boundaries=get_quantile_based_boundaries(training_examples[feature], buckets_count))
        if feature == 'longitude':
            bucket_logitude = bucketized_feature
        if feature == 'latitude':
            bucket_latitude = bucketized_feature
        feature_columns.add(bucketized_feature)

    # build the feature cross will only work with the california housing example
    feature_columns.add(tf.feature_column.crossed_column([bucket_latitude, bucket_logitude], 1000))

    return feature_columns