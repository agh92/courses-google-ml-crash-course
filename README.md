# google_ml_crash_course
Code and exercises from [Google´s Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)

## Steps to create a model

1. Load data as pandas [`DataFrame`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
2. Define the input features as pandas [`Series`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html)
3. Define the targets as pandas `Series`
4. Define an optimizer to reduce loss
5. Define the type of the model as [`tf.estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator)
6. Define the input function that provides de estimator with a tuple of features and labels
7. Train the model
8. Make Predictions
9. Calculate loss

## Clipping Data

1. Analyse distribution to find outliers
2. Use [`Series.apply`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html#pandas.Series.apply) to replace or cut outliers out 
3. Repeat steps 7 - 9 from above
