import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import sys
sys.path.append('../')
from utils import transforms
from matplotlib import pyplot as plt


US_MAP = 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_1_states_provinces_shp.geojson'


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