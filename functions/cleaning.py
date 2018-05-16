import math


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
        min_latitude = math.trunc(source_df[feature].min())
        max_latitude = int(math.ceil(source_df[feature].max()))

        # LATITUDE_RANGES = zip(xrange(32, 44), xrange(33, 45))
        ranges = zip(xrange(min_latitude, max_latitude), xrange(min_latitude + 1, max_latitude + 1))

    selected_examples = source_df.copy()
    for r in ranges:
        selected_examples[(feature + "_%d_to_%d") % r] = source_df[feature].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples
