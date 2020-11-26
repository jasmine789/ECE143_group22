import shapely
from shapely.geometry import MultiPolygon
from shapely import affinity
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
        pd_df, geometry=gpd.points_from_xy(pd_df.longitude, pd_df.latitude),
        crs="EPSG:4326")

    return gdf_point


def geometry_transform(gdf, row=0, offset=[0, 0], rotate=0, scale=[0, 0]):
    """
    Perform translation on the shapely object and return the new object
    :param df: given geopandas dataframe
    :param attr: given row that need to be translated
    :param offset: given list of offset corresponds to x, y axis respectively
    :param rotate: rotate angle
    :param scale: given scale factor corresponds to x, y respectively
    :return:

    currently im unhappy about the way
    """
    assert isinstance(row, int) and row in list(gdf.index)
    assert isinstance(offset, (list, tuple))
    assert isinstance(scale, (list, tuple))

    source = gdf['geometry'][row]
    source = affinity.translate(source, *offset)
    source = affinity.rotate(source, rotate)
    source = affinity.scale(source, *scale)

    gdf['geometry'][row] = source

    return gdf


if __name__ == '__main__':
    from configs import *
    from matplotlib import pyplot as plt
    df = gpd.read_file(US_MAP)

    df = geometry_transform(df, 50, [20, -30], 0, [0.3, 0.3])
    df = geometry_transform(df, 3, [20, 20], 0, [0.5, 0.5])

    df.plot()

    plt.show()





