import shapely
import pandas as pd
import geopandas as gpd


def pd2Point(pd_df):
    """
    Function that maps panda frame locations to geopandas Point object
    :param pd_df: panda data frame
    :return:
    """
    assert isinstance(pd_df, pd.DataFrame)
    gdf_point = gpd.GeoDataFrame(
        pd_df, geometry=gpd.points_from_xy(pd_df.longitude, pd_df.latitude))

    return gdf_point



