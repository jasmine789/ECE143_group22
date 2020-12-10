import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

US_MAP = 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_1_states_provinces_shp.geojson'


def plot_intchoropleth_map(pd_df, attr, us_only=True, **kwargs):
    """
    Function that plots interactive choropleth map
    :param pd_df: input pandas dataframe
    :param attr: given attributes we are interested in
    :param us_only: whether to plot the us only map or not
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


if __name__ == '__main__':
    data = pd.read_csv('../MergeCommon_loc_disposition.csv', engine='python')
    plot_intbubble_map(data, hovertext=data['city'])