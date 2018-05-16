
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


def plot_loss_over_periods(loss_periods_dict):
    """

    :param loss_periods_dict:
    :return:
    """
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    for key, value in loss_periods_dict.items():
        plt.plot(value, label=key)
    plt.legend()


def plot_sample(sample, my_feature, my_label, periods):
    """

    :param sample:
    :param my_feature:
    :param my_label:
    :param periods:
    :return:
    """
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    plt.scatter(sample[my_feature], sample[my_label])
    return [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]


def plot_linear_model(sample, my_label, linear_regressor, input_feature, color):
    """

    :param sample:
    :param my_label:
    :param linear_regressor:
    :param input_feature:
    :param color:
    :return:
    """
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])

    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[input_feature].max()),
                           sample[input_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=color)