from matplotlib import pyplot as plt
import collections
import plotly.express as px
from configs import *
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import numpy as np
import pandas as pd
import geopandas as gpd
import transforms
from pywaffle import Waffle


def plot_histogram(keys, values, xlabel="x", ylabel="y", title='',sort=False, size=(20, 10), **kwargs):
    """
    Function that plot the given data as histograms
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

    fig = plt.figure(figsize=size)

    # set bar color
    offset = 1 # this offset is set to avoid pure white bars
    norm = plt.Normalize(min(values)-offset, max(values)+offset)
    norm_y = norm(values)
    map_vir = matplotlib.cm.get_cmap(name='hot')

    plt.bar(np.arange(L), values, color=map_vir(norm_y), **kwargs)

    # add numerical tags
    for (x, y) in zip(np.arange(L), values):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.xticks(np.arange(L), keys, rotation=-20, fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title(title, fontweight='bold')

    plt.show()


def plot_choropleth_map(pd_df, attr, axis_on=False, size=(20, 10), **kwargs):
    """
    Function that plot the choropleth map with the assist of geopands, notice that
    :param gpd_df: geopandas dataframe
    :param attr: attributes we are interested in
    :param axis_on: whether to show the axis or not
    :param size: figure size
    :param **kwargs:
    {
    }
    :return:
    """
    assert isinstance(pd_df, pd.DataFrame)
    assert isinstance(attr, str) and attr is not None
    assert isinstance(axis_on, bool)
    assert isinstance(size, (list, tuple))

    column_map = dict(zip(pd_df['State Abbreviation'].values, pd_df[attr].values))

    # load the default US map
    gpd_df = gpd.read_file(US_MAP)

    # add the data column into the geodataframe
    gpd_df[attr] = gpd_df.apply(lambda x: column_map[x['postal']], axis=1)

    fig, ax = plt.subplots(figsize=size)
    gpd_df.plot(ax=ax, column=gpd_df[attr], **kwargs)

    # plot the state abbreviation
    for i in range(len(gpd_df)):
        state_abbr, loc = gpd_df.iloc[i, :]['postal'], gpd_df.iloc[i, :]['geometry'].centroid
        x, y = loc.coords[:][0]
        plt.text(x, y, state_abbr, fontweight='black', fontsize=20, fontstyle='italic', ha='center')
    ax.axis(axis_on)

    plt.show()


def plot_bubble_map(pd_point, axis_on=False, size=(20, 10), **kwargs):
    """
    Function that plot the bubble map with the assist of geopands
    :param pd_point: pandas dataframe with longitude and latitude
    :param axis_on: whether to show the axis or not
    :param size: figure size
    :param **kwargs:
    {
    }
    :return:
    """
    assert isinstance(pd_point, pd.DataFrame)
    assert isinstance(axis_on, bool)
    assert isinstance(size, (list, tuple))

    # plot the default us map
    gpd_df = gpd.read_file(US_MAP)

    # plot the state abbreviation
    for i in range(len(gpd_df)):
        state_abbr, loc = gpd_df.iloc[i, :]['postal'], gpd_df.iloc[i, :]['geometry'].centroid
        x, y = loc.coords[:][0]
        plt.text(x, y, state_abbr, fontweight='black', fontsize=20, fontstyle='italic', ha='center')

    fig, ax = plt.subplots(figsize=size)
    gpd_df.plot(ax=ax, color='lightgray', edgecolor='grey', linewidth=2, **kwargs)
    ax.axis(axis_on)

    # plot the bubble point
    assert 'longitude' in pd_point and 'latitude' in pd_point
    gpd_point = transforms.pd2Point(pd_point)
    gpd_point.plot(ax=ax, color='#07424A', markersize=2, alpha=0.7, categorical=False, legend=True)

    plt.show()


def plot_pie_chart(df, attr, label, th, **kwargs):
    """
    Plot pie chart using the given pandas dataframe
    :param df: given pandas dataframe
    :param attr: given attributes we are interested in
    :param label: plot labels
    :param th: given threshold that the values below it will be assigned as 'Others'
    :param kwargs:
    :return:
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(attr, str) and attr in df
    assert isinstance(label, str) and label in df

    df.loc[df[attr] < th, label] = 'Others'
    fig = px.pie(df, values=attr, names=label, **kwargs)
    fig.show()


def plot_wordcloud(df, attr, size=(10, 10), stopwords=set(), **kwargs):
    """
    Plot a wordcloud figure with given pandas dataframe and the attr we are interested in
    :param df: pandas dataframe
    :param attr: given attributes
    :param size: figure size
    :param stopwords: stopwords set, if not specified, use the default stopwords in wordcloud
    :param kwargs:
    :return:
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(attr, str) and attr in df
    assert isinstance(size, (list, tuple))
    assert isinstance(stopwords, set)

    if not stopwords:
        stopwords = set(STOPWORDS)

    freq = df[attr].value_counts().to_dict()

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          **kwargs).generate_from_frequencies(freq)

    # plot the WordCloud image
    fig, ax = plt.subplots(figsize=size, facecolor=None)
    ax.imshow(wordcloud)
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.show()


def plot_scatter_2D(df, x_attr, y_attr, ps, xlabel='x', ylabel='y', size=(20, 10), **kwargs):
    """
    Function that plots 2d scatter figure based on the given x,y data
    :param df: given pandas data frame
    :param x_attr: given attributes on the x_axis
    :param y_attr: given attributes on the y_axis
    :param ps: given point size, normally this depends on the population
    :param xlabel: given x label
    :param ylabel: given y label
    :param size: figure size
    :param kwargs:
    :return:
    """
    assert isinstance(x_attr, str)
    assert isinstance(y_attr, str)
    assert x_attr in df and y_attr in df
    assert isinstance(xlabel, str) and isinstance(ylabel, str)
    assert isinstance(size, (list, tuple))

    fig = px.scatter(df,
                     x=x_attr,
                     y=y_attr,
                     color='State Abbreviation',
                     size=ps,
                     size_max=60,
                     width=900,
                     height=400,
                     title='Crime Rate/Violence Rate',
                     )
    fig.show()


def plot_waffle(pd_df, attr, title="", rows=15, cols=20, **kwargs):
    """
    Plot waffle plots by specifying rows and cols with pandas dataframe
    :param pd_df: pandas dataframe
    :param rows:
    :param cols:
    :param values:
    :return:
    """
    assert isinstance(pd_df, pd.DataFrame)
    assert isinstance(attr, str) and attr in pd_df
    assert isinstance(rows, int) and rows > 1
    assert isinstance(cols, int) and cols > 1

    # count the attributes and do a simple preprocesing
    counts = pd_df[attr].value_counts().to_dict()
    values = collections.defaultdict(int)
    for key in counts:
        if 'Convicted' in key:
            values['Convicted'] += counts[key]
        elif 'Charged' in key:
            values['Charged'] += counts[key]
        else:
            values['No known charges'] += counts[key]

    fig = plt.figure(
        FigureClass=Waffle,
        rows=rows,
        columns=cols,
        values=values,
        icons='male',
        font_size=15,
        interval_ratio_x=0.4,
        interval_ratio_y=1,
        legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
        **kwargs
    )
    plt.title(title)
    fig.show()


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

    fig = plt.figure(figsize=size)

    plt.bar(x, values, weights, color=['orangered']+['silver']*(len(x)-1))

    # add text on top of the bar and middle
    for _x, _y, k, c in zip(x, values, keys, count):
        plt.text(_x, _y + 0.05, k.split()[0], ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.text(_x, _y/2, '%d killed'% c, ha='center', va='bottom', fontsize=12)

    # plt.axis('off')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.xticks(x, ['39.7M', '57.5M', '197.2M', '17.4M'], fontsize=12)

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../2013-2019 Killings by State and crime rate.csv')
    pop = ['Black Population', 'Hispanic Population', 'Asian Population', 'White Population']
    count = ['# Black people killed', '# Hispanic people killed', '# Asian people killed', '# White people killed']

    pops = np.array([data[key].sum() for key in pop])
    weights = pops / pops.sum()
    y = np.array([data[key].sum() for key in count]) / pops * 10**6
    killings = np.array([data[key].sum() for key in count])

    plot_weighted_bar(pop, y, weights, killings,
                      xlabel='Part of U.S population',
                      ylabel='Killings per million',
                      title='Racial disparity in killing rate',
                      sort=True)


