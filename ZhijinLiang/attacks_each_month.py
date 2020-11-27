import pandas as pd
from collections import defaultdict
import holoviews as hv
import matplotlib.pyplot as plt
from operator import add
import numpy as np
hv.extension('bokeh')

def attack_over_year_Plot(fileName, button = 'year', outputName = 'attack_over_year.html'):
    """
    This function is to plot the number of attacks over months for each year
    :param fileName: A given file in csv format, it should contains a 'date' column
    :param button: the interactive button, suppors 'year' or 'month'
    :param outputName: the path and name to store the genreated interactive plot(in .html format).
    :return: a plot that will be saved in .html
    """
    assert (button == 'year' or button == 'month')
    
    attacks_pd,_ = processData_attack(fileName)
    hvData = hv.Dataset(data=attacks_pd, kdims=['year', 'month'])
    
    if button == 'year':
        fig = hvData.to(hv.Curve, 'month', 'NO. attacks', groupby= button).options(height=200)
    else:
        fig = hvData.to(hv.Curve, 'year', 'NO. attacks', groupby= button).options(height=200)

    hv.save(fig, outputName)
    return fig

def total_attackes_over_year_Simple(fileName, outputName = 'total_attack_over_year_bar.png'):
    """
    This function is to generate a bar plot of the total number of attacks over year
    :param fileName: A given file in csv format, it should contains a 'date' column
    :param outputName: the path and name to store the genreated interactive plot(in .html format).
    :return: a bar plot that will be saved in the given outputName
    """
    _, attacksPerYear = processData_attack(fileName)
    yearList = ['2013', '2014', '2015','2016','2017','2018','2019','2020']
    totalAttacksPerYear = []
    for year in attacksPerYear.keys():
        curSum = 0
        for month in attacksPerYear[year].keys():
            curSum = curSum + attacksPerYear[year][month]
        totalAttacksPerYear.append(curSum)

    # Generate plot
    bars = plt.bar(yearList, totalAttacksPerYear)
    plt.title('The number of total Attacks over Year')
    plt.xlabel('year (2020 up to Oct/2020)')
    plt.ylabel('Attacks Num')

    # add counts above the bar plot
    for rect in bars:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, h, '%d' % int(h), ha='center', va='bottom')

    plt.savefig(outputName)
    plt.show()


def total_attackes_over_year_Advanced(fileName, colorMap = "viridis", outputName = 'total_attack_over_year_bar_advanced.png'):
    """
    This function is to generate a bar plot of the total number of attacks over year(show month as well)
    :param fileName: A given file in csv format, it should contains a 'date' column
    :param colorMap: the cmap that will be used in plot
    :param outputName: the path and name to store the genreated interactive plot(in .html format).
    :return: a bar plot that will be saved in the given outputName
    """
    _, attacksPerYear = processData_attack(fileName)
    yearList = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
    monthList = ['01','02','03','04','05','06','07','08','09','10','11','12']
    totalAttacksPerMonth = defaultdict(list) # {Mont:[att in 2013, att in 2014, ...],...}
    for month in monthList:
        for year in attacksPerYear.keys():
            if month in attacksPerYear[year]:
                totalAttacksPerMonth[month].append(attacksPerYear[year][month])
            else:
                totalAttacksPerMonth[month].append(0)

    # # accumulate the attacks per month to make plot
    totalAttacksPerMonthAcc = defaultdict(list) # {Mont:[att in 2013, att in 2014, ...],...}, with accumulation
    totalAttacksPerMonthAcc['01'] = totalAttacksPerMonth['01']
    for i in range(1, len(monthList)):
        lastMonth = monthList[i-1]
        month = monthList[i]
        totalAttacksPerMonthAcc[month] = list(map(add, totalAttacksPerMonthAcc[lastMonth], totalAttacksPerMonth[month]))

    # Generate plot
    yList = np.arange(15,15+12*2,2)
    rescale = lambda y: (y-np.min(yList)) / (np.max(yList) - np.min(yList))
    colorSet = [plt.get_cmap(colorMap)(rescale(y)) for y in yList]
    bars = [plt.bar(yearList, totalAttacksPerMonth['01'], color = colorSet[0])]
    for i in range(1, len(monthList)):
        lastMonth = monthList[i - 1]
        month = monthList[i]
        bars.append(plt.bar(yearList, totalAttacksPerMonth[month], bottom=totalAttacksPerMonthAcc[lastMonth],
                            color=colorSet[i]))

    plt.title('The number of total Attacks over Year')
    plt.xlabel('year (2020 up to Oct/2020)')
    plt.ylabel('Attacks Num')

    # add counts above the bar plot
    count = 0
    for rect in bars[-1]:
        h = totalAttacksPerMonthAcc['12'][count]
        plt.text(rect.get_x() + rect.get_width() / 2.0, h, '%d' % int(h), ha='center', va='bottom')
        count = count + 1

    plt.savefig(outputName)
    plt.show()

def processData_attack(fileName):
    """
    This dataset is used to process a dataset, return a pd.DataFrame in : Year|month|No.attcks
    :param fileName: A given file in csv format, it should contains a 'date' column
    :return: pd.DataFrame and a nested dictionary({year:{month:attacks, month:attacks,...},...}
    """
    dataset = pd.read_csv(fileName)
    attacksPerYear = {}  # key: year(str), value: a dictionary stores attachePerMonth dictionary
    yearList = ['2013', '2014', '2015','2016','2017','2018','2019','2020']
    for year in yearList:
        attacksPerYear[year] = count_attacks_each_month(year, dataset)

    # Create a pd.DataFrame: Year|month|No.attacks
    attacks_pd = pd.DataFrame()
    year_list_inPd = []
    month_list_inPd = []
    attackNo_list_inPd = []
    for year in attacksPerYear.keys():
        for month in attacksPerYear[year].keys():
            year_list_inPd.append(year)
            month_list_inPd.append(month)
            attackNo_list_inPd.append(attacksPerYear[year][month])

    attacks_pd['year'] = year_list_inPd
    attacks_pd['month'] = month_list_inPd
    attacks_pd['NO. attacks'] = attackNo_list_inPd

    return attacks_pd, attacksPerYear


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
    # fig = attack_over_year_Plot(fileName)
    # total_attackes_over_year_Simple(fileName)
    total_attackes_over_year_Advanced(fileName)




