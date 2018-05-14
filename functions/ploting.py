
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


def loss_over_periods(data, show=True):
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    for key, value in data.items():
        plt.plot(value, label=key)
    plt.legend()
    if show:
        plt.show()

def state_line(sample, my_feature, my_label, periods):
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    plt.scatter(sample[my_feature], sample[my_label])
    return [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]