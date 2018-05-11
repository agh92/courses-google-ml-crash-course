# google_ml_crash_course
Code and exercises of the Machine Learning Crash Course

## Steps to create a model

1. Load data as pandas `DataFrame`
2. Define the input features as pandas `Series`
3. Define the targets as pandas `Series`
4. Define an optimizer to reduce loss
5. Define the type of the model as `tf.estimator`
6. Define the input function that provides de estimator with a tuple of features and labels
7. Train the model
8. Make Predictions
9. Calculate loss