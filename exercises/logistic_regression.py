import pandas as pd
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from sklearn import metrics
from functions import data_processing as dp
from functions import training as tr

#########
# SETUP #
#########
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = dp.load_data_frame_from_csv('../data/california_housing_train.csv')
training_examples, training_targets, validation_examples, validation_targets = dp.test_and_validation(
    california_housing_dataframe, binary=True)

# Double-check that we've done the right thing.
# print "Training examples summary:"
# display.display(training_examples.describe())
# print "Validation examples summary:"
# display.display(validation_examples.describe())
#
# print "Training targets summary:"
# display.display(training_targets.describe())
# print "Validation targets summary:"
# display.display(validation_targets.describe())


# used to predict a value between 0 and 1 and would be interpreted as a probability
# linear_regressor = tr.train_model_all_features(
#     learning_rate=0.000001,
#     steps=200,
#     batch_size=20,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets,
#     my_target='median_house_value_is_high')

# All metrics improve at the same time, so our loss metric is a good proxy for both AUC and accuracy.
# Notice how it takes many, many more iterations just to squeeze a few more units of AUC. This commonly happens.
# But often even this small gain is worth the costs.
linear_classifier = tr.train_linear_classifier_model(
    learning_rate=0.000004,
    steps=50000,
    batch_size=200,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

predict_validation_input_fn = lambda: dp.my_input_fn(validation_examples,
                                                      validation_targets['median_house_value_is_high'],
                                                      num_epochs=1,
                                                      shuffle=False)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print "AUC on the validation set: %0.2f" % evaluation_metrics['auc']
print "Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy']

# validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
#
# false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
#     validation_targets, validation_probabilities)
# plt.plot(false_positive_rate, true_positive_rate, label="our model")
# plt.plot([0, 1], [0, 1], label="random classifier")
# _ = plt.legend(loc=2)
# plt.show()

