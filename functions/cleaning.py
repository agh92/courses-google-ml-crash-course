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
            numeric_feature, boundaries=get_quantile_based_boundaries(training_examples[feature], feature_bucktes_count[feature]))
        feature_bucket[feature] = bucketized_feature
        feature_columns.add(bucketized_feature)

    for features_list in cross_features:
        buckets = []
        for feature in features_list:
            buckets.append(feature_bucket[feature])
        feature_columns.add(tf.feature_column.crossed_column(buckets, 1000))

    return feature_columns
