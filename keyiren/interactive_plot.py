from urllib.request import urlopen
from configs import *
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go


def plot_intchoropleth_map(pd_df, attr, us_only=True, **kwargs):
    """
    Function that plots interactive choropleth map
    :param pd_df:
    :param attr:
    :return:
    """
    assert isinstance(pd_df, pd.DataFrame)

    column_map = dict(zip(pd_df['State Abbreviation'].values, pd_df[attr].values))

    gpd_df = gpd.read_file(US_MAP)
    gpd_df[attr] = gpd_df.apply(lambda x: column_map[x['postal']], axis=1)

    if not us_only:
        fig = px.choropleth_mapbox(gpd_df,
                                   geojson=gpd_df.geometry,
                                   locations=gpd_df.index,
                                   color=attr,
                                   color_continuous_scale="reds",
                                   hover_data=[attr, 'postal', ''],
                                   **kwargs
                                   )
    else:
        fig = px.choropleth(gpd_df,
                            geojson=gpd_df.geometry,
                            locations=gpd_df.index,
                            color=attr,
                            color_continuous_scale="reds",
                            hover_data=[attr, 'postal'],
                            scope='usa',
                            **kwargs
                            )

    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom=3,
                      mapbox_center={"lat": 37.0902, "lon": -95.7129},
                      )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_geos(
        subunitcolor="white",
    )

    fig.show()


def plot_intbubble_map(pd_df, axis_on=False, size=(20, 10), **kwargs):
    """
    Function that plot the bubble map with the assist of geopands
    :param pd_df: pandas dataframe with longitude and latitude
    :param axis_on: whether to show the axis or not
    :param size: figure size
    :param **kwargs:
    {
    }
    :return:
    """
    assert isinstance(pd_df, pd.DataFrame)
    assert 'longitude' in pd_df and 'latitude' in pd_df
    assert isinstance(axis_on, bool)
    assert isinstance(size, (list, tuple))

    fig = go.Figure(data=go.Scattergeo(
        lon=pd_df['longitude'],
        lat=pd_df['latitude'],
        geojson=US_MAP,
        marker=dict(
            symbol="x",
            color='#07424A',
            reversescale=True,
            opacity=0.7,
            size=4,
        ),
        **kwargs,
    ))

    fig.update_layout(
        title='Police Violence Occurrence in the U.S',
        geo_scope='usa',
    )
    fig.update_geos(
        subunitcolor="white",
    )
    fig.show()


def plot_weighted_bar(keys, values, weights, xlabel="x", ylabel="y", sort=False, size=(20, 10), **kwargs):
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

    fig = go.Figure(data=[go.Bar(
        x=keys,
        y=values,
        width=weights,
        hovertext=[],

    )])

    fig.show()


if __name__ == '__main__':
    data = pd.read_csv('../MergeCommon_loc_disposition.csv', engine='python')
    plot_intbubble_map(data, hovertext=data['city'])
    # pop = ['Black Population', 'Hispanic Population', 'Asian Population', 'White Population', 'Other Population']
    # count = ['# Black people killed', '# Hispanic people killed', '# Asian people killed', '# White people killed',
    #          '# Unknown Race people killed']
    #
    # pops = np.array([data[key].sum() for key in pop])
    # x = [2, 5, 8, 12, 16]
    # weights = [2, 3.4, 1, 7.4, 0.4]
    # y = np.array([data[key].sum() for key in count]) / pops
    # plot_weighted_bar(x, y, weights)