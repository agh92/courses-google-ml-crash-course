import pandas as pd


def binning_feature(source_df, ranges, feature):
    selected_examples = source_df.copy()
    for r in ranges:
        selected_examples[(feature + "_%d_to_%d") % r] = source_df[feature].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples
