import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def plot_bar(keys, values, xlabel="x", ylabel="y", title='', sort=False, size=(30, 15), **kwargs):
    """
    Function that plot the given data as bar plots
    :param keys: given key data on the X axis
    :param values: given value data on the Y axis
    :param xlabel: label of x axis
    :param ylabel: label of y axis
    :param title: title of the plot
    :param sort: flag indicates whether the data is sorted
    :return: None
    """
    assert isinstance(keys, (pd.Series, np.ndarray, tuple, list))
    assert isinstance(values, (pd.Series, np.ndarray, tuple, list))
    assert isinstance(xlabel, str) and isinstance(ylabel, str)
    assert isinstance(sort, bool)
    assert isinstance(size, (list, tuple))

    if sort:
        keys, values = zip(*sorted(zip(keys, values), key=lambda x: x[1], reverse=True))

    L = len(values)

    fig, ax = plt.subplots(figsize=size)

    # set bar color
    offset = 3 # this offset is set to avoid pure white bars
    norm = plt.Normalize(min(values)-offset, max(values)+offset)
    norm_y = 1-norm(values)
    map_vir = matplotlib.cm.get_cmap(name='hot')

    plt.bar(np.arange(L), values, color=map_vir(norm_y), **kwargs)

    # add numerical tags
    x_idx = [0, 1, 2, L-3, L-2, L-1]
    for x in x_idx:
        plt.text(x, values[x] + 0.05, '%.2f' % values[x], ha='center', va='bottom', fontsize=15, fontweight='bold')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(np.arange(L), keys, rotation=-20, fontsize=20)
    plt.yticks(fontsize=20)

    plt.title(title, fontsize=30, fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()


def plot_weighted_bar(keys, values, weights, count, xlabel="x", ylabel="y", title="", sort=False, size=(20, 10), **kwargs):
    """
    Plot interactive weighted bar plot using plotly
    Function that plot the given data as histograms
    :param keys: given key data on the X axis
    :param values: given value data on the Y axis
    :param weights: along which attributes do we specify the bar weights
    :param xlabel: label of x axis
    :param ylabel: label of y axis
    :param sort: flag indicates whether the data is sorted
    :param size: default plot size
    :return: None
    :return:
    """
    assert isinstance(keys, (pd.Series, np.ndarray, tuple, list))
    assert isinstance(values, (pd.Series, np.ndarray, tuple, list))
    assert isinstance(xlabel, str) and isinstance(ylabel, str)
    assert isinstance(sort, bool)
    assert isinstance(size, (list, tuple))

    interval = 0.01

    # sort if needed
    if sort:
        values, weights, keys, count = zip(*sorted(zip(values, weights, keys, count), reverse=True))

    # transform weights into x coordinates
    x = []
    for i, w in enumerate(weights):
        if not x:
            x.append(w/2)
        else:
            x.append(x[-1]+weights[i-1]/2+interval+w/2)

    fig, ax = plt.subplots(figsize=size)

    plt.bar(x, values, weights, color=['orangered']+['silver']*(len(x)-1))

    # add text on top of the bar and middle
    for _x, _y, k, c in zip(x, values, keys, count):
        plt.text(_x, _y + 0.05, k.split()[0], ha='center', va='bottom', fontsize=15, fontweight='bold')
        plt.text(_x, _y/2, '%d killed'% c, ha='center', va='bottom', fontsize=15)

    # plt.axis('off')

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.xticks(x, ['39.7M', '57.5M', '197.2M', '17.4M'], fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()