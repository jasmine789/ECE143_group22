import pandas as pd
import collections
from pywaffle import Waffle
from matplotlib import pyplot as plt


def plot_waffle(pd_df, attr, rows=15, cols=20, **kwargs):
    """
    Plot waffle plots by specifying rows and cols with pandas dataframe
    :param pd_df: pandas dataframe
    :param attr: given attributes  we are interested in
    :param rows: number of rows in the waffle plot
    :param cols: number of columns in the waffle plot
    :return:
    """
    assert isinstance(pd_df, pd.DataFrame)
    assert isinstance(attr, str) and attr in pd_df
    assert isinstance(rows, int) and rows > 1
    assert isinstance(cols, int) and cols > 1

    # count the attributes and do a simple preprocess
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
        font_size=16,
        interval_ratio_x=2,
        interval_ratio_y=2,
        legend={'loc': 'lower left',
               'bbox_to_anchor': (0, -0.2),
               'ncol': 3,
               'framealpha': 0,
               'fontsize': 12
               },
        **kwargs
    )

    fig.show()