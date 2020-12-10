import pandas as pd
import geopandas as gpd
import plotly.express as px
from matplotlib import pyplot as plt

US_MAP = 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_1_states_provinces_shp.geojson'


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