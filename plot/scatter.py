import plotly.express as px


def plot_scatter_2D(df, x_attr, y_attr, ps=None, xlabel='x', ylabel='y', size=(20, 10), **kwargs):
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
                     # size=ps,
                     width=900,
                     height=400,
                     title='Crime Rate/Violence Rate',
                     trendline="ols",
                     )
    fig.show()