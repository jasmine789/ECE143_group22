import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt


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
                          relative_scaling=0.3,
                          **kwargs).generate_from_frequencies(freq)

    # plot the WordCloud image
    fig, ax = plt.subplots(figsize=size, facecolor=None)
    ax.imshow(wordcloud)
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.show()