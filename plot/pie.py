import plotly.express as px
import pandas as pd


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