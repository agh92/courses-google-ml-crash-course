import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from sklearn.ensemble._gradient_boosting import np_bool
from tensorflow.python.data import Dataset
import miscelanius.data_processing as dp

#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("../data/california_housing_train.csv", sep=",")
##########
# TASK 3 #
##########
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

training_set = dp.preprocess_features(california_housing_dataframe.head(12000))
print training_set.describe()
training_targets = dp.preprocess_targets(california_housing_dataframe.head(12000))
#print training_targets.describe()

validation_set = dp.preprocess_features(california_housing_dataframe.head(5000))
print validation_set.describe()
validation_targets = dp.preprocess_targets(california_housing_dataframe.head(5000))
#print validation_set.describe()

##########
# TASK 1 #
##########
# Does the data fulfills the expectations?
# Are there odd things?

##########
# TASK 2 #
##########
# The key thing to notice is that for any given feature or column, the distribution of values between the train and
# validation splits should be roughly equal !!! -> Fix in Task 3
plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_set["longitude"],
            validation_set["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_set["longitude"],
            training_set["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()
plt.show()
