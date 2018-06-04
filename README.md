# google_ml_crash_course
Code and exercises from [Google´s Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)

Directories:
* data: Contains training and test data for the examples
* exercises: Contains all programming exercises of the course
* functions: Python package containing code that is reusable across the examples.

## Steps to create a model

1. Load data as pandas [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
2. Define the input features as pandas [`Series`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html)
3. Define the targets as pandas `Series`
4. Define an optimizer from [`tensorflow.train`](https://www.tensorflow.org/api_docs/python/tf/train) to reduce loss
5. Define the type of the model as [`tf.estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator)
6. Define the input function that provides de estimator with a tuple of features and labels
7. Train the model
8. Make Predictions
9. Compute loss using [`tf.metrics`](https://www.tensorflow.org/api_docs/python/tf/metrics) or [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) helper functions

## Clipping Data

1. Analyse distribution to find outliers
2. Use [`Series.apply`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html#pandas.Series.apply) to replace or cut outliers out 
3. Repeat steps 7 - 9 from above

## Validation Strategy

1. Load a **train data set**
2. Load a **validation data set**
3. Train the model with the **train set** and check the loss against the **validation set**
4. Load a **test data set**
5. Check the loss against the **test set** and not validation set 
