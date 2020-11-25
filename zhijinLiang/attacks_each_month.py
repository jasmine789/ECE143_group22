import pandas as pd
from collections import defaultdict
import holoviews as hv
hv.extension('bokeh')

def attack_over_year_Plot(fileName, outputName = 'attack_over_year.html'):
    """
    This function is plot the number of attacks over months for each year
    :param fileName: A given file in csv format, it should contains a 'date' column
    :return: a plot that will be samed in .html
    """
    attacks_pd = processData_attack(fileName)
    hvData = hv.Dataset(data=attacks_pd, kdims=['year', 'month'])

    fig = hvData.to(hv.Curve, 'month', 'NO. attacks', groupby='year').options(height=200)
    hv.save(fig, outputName)

def processData_attack(fileName):
    """
    This dataset is used to process a dataset, return a pd.DataFrame in : Year|month|No.attcks
    :param fileName: A given file in csv format, it should contains a 'date' column
    :return: pd.DataFrame
    """
    dataset = pd.read_csv(fileName)
    attacksPearYear = {}  # key: year(str), value: a dictionary stores attachePerMonth dictionary
    yearList = ['2013', '2014', '2015','2016','2017','2018','2019','2020']
    for year in yearList:
        attacksPearYear[year] = count_attacks_each_month(year, dataset)

    # Create a pd.DataFrame: Year|month|No.attcks
    attacks_pd = pd.DataFrame()
    year_list_inPd = []
    month_list_inPd = []
    attackNo_list_inPd = []
    for year in attacksPearYear.keys():
        for month in attacksPearYear[year].keys():
            year_list_inPd.append(year)
            month_list_inPd.append(month)
            attackNo_list_inPd.append(attacksPearYear[year][month])

    attacks_pd['year'] = year_list_inPd
    attacks_pd['month'] = month_list_inPd
    attacks_pd['NO. attacks'] = attackNo_list_inPd

    return attacks_pd


def count_attacks_each_month(GivenYear, data):
    """
    This function is used to count how many attacks will happend on each month for the given year
    :param GivenYear: The provided year number in string, e.g. 2012
    :param data: dataset, a pd.DataFrame, it should contains a 'date' column
    :return: a dictionary (key: month, value: the number of attacks)
    """
    assert isinstance(GivenYear, str)
    assert isinstance(data, pd.DataFrame)

    attacksPerMonth = defaultdict(int)
    for row in data['date']:
        rowVal = row.split('-')
        year, month, day = rowVal[0], rowVal[1], rowVal[2]
        if year == GivenYear:
            attacksPerMonth[month] = attacksPerMonth[month] + 1
        else:
            pass

    attacksPerMonthSort = dict(sorted(attacksPerMonth.items(), key = lambda kv: kv[0]))
    return attacksPerMonthSort


if __name__ == '__main__':
    # load csv files
    fileName = 'MergeCommon_final.csv'
    attack_over_year_Plot(fileName)





