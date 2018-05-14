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


def preprocess_features(data_frame):
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
    processed_features["rooms_per_person"] = (
            data_frame["total_rooms"] /
            data_frame["population"])

    return  processed_features


def preprocess_targets(data_frame):
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = (
            data_frame['median_house_value'] / 1000.0 )
    return output_targets