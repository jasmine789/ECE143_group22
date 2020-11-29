# This file contains the age distribution plot(including data processing and plot)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme()

def age_distribution(fileName, outputName = 'age_distribution.png'):
    """
    Generate age distribution based on all data (I drop the info if data's age information is missing,
    there are not too much missing information)
    :param fileName: A given file in csv format, it should contains a 'age' column
    :param outputName: the path and name to store the plot
    :return: age distribution plot
    """
    data = pd.read_csv(fileName, engine='python')
    fig_sns = sns.distplot(data['age'], hist=True, kde=True, hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    fig_sns.set_title('Age Distribution of attacks (2013-2020)')
    fig_sns.set_ylabel('Frequency')
    fig_sns.get_figure().savefig(outputName)
    return fig_sns

def age_distribution_races(fileName, outputName = 'age_distribution_race.png'):
    """
    Generate age distribution based on all data, for different races (I remove the data entry if the race information
    is not provided, but there are not too much missing info)
    :param fileName: A given file in csv format, it should contains a 'age' column
    :param outputName: the path and name to store the plot
    :return: age distribution plot
    """
    data = pd.read_csv(fileName, engine='python')
    # check all possible type of races
    # data['race'].unique()

    age_A = data[data['race'] == 'A']['age'].values
    age_W = data[data['race'] == 'W']['age'].values
    age_H = data[data['race'] == 'H']['age'].values
    age_B = data[data['race'] == 'B']['age'].values
    age_O = data[data['race'] == 'O']['age'].values
    age_N = data[data['race'] == 'N']['age'].values

    # Generate plots
    fig, axs = plt.subplots(nrows=2, figsize= (15,10))

    # the first plot is distribution plot
    sns.distplot(age_A, hist = False, ax=axs[0])
    sns.distplot(age_W, hist = False, ax=axs[0])
    sns.distplot(age_H, hist= False, ax=axs[0])
    sns.distplot(age_B, hist = False, ax=axs[0])
    sns.distplot(age_O, hist= False, ax=axs[0])
    fig_sns = sns.distplot(age_N, hist= False, ax=axs[0])
    fig_sns.set_title('Age Distribution of attacks for different races (2013-2020)')
    fig_sns.set_ylabel('Frequency')
    fig_sns.set_xlabel('age')
    fig_sns.legend(labels = ['Asian', 'White', 'Hispanic', 'Black', 'Pacific Islander', 'Native American'])

    # The 2nd plot is a box plot - need to modify the race label here
    # newRace.replace(to_replace = 'A',value = 'Asian')
    # newRace.replace('W', 'White')
    # newRace.replace('H', 'Hispanic')
    # newRace.replace('B', 'Black')
    # newRace.replace('O', 'Pacific Islander')
    # newRace.replace('N', 'Native American')
    # data['newRace'] = newRace
    data['race'] = data['race'].replace(['A'], 'Asian')
    data['race'] = data['race'].replace(['W'], 'White')
    data['race'] = data['race'].replace(['B'], 'Black')
    data['race'] = data['race'].replace(['O'], 'Pacific Islander')
    data['race'] = data['race'].replace(['N'], 'Native American')
    data['race'] = data['race'].replace(['H'], 'Hispanic')

    sns.boxplot(x="age", y="race", data=data, orient ='h', ax = axs[1])

    # store the distribution
    fig_sns.get_figure().savefig(outputName)
    return fig_sns

def age_distribution_gender(fileName, outputName = 'age_distribution_gender.png'):
    """
    Generate age distribution based on all data, for different races (I remove the data entry if the race information
    is not provided, but there are not too much missing info)
    :param fileName: A given file in csv format, it should contains a 'age' column
    :param outputName: the path and name to store the plot
    :return: age distribution plot
    """
    data = pd.read_csv(fileName, engine='python')
    age_Female = data[data['gender'] == 'F']['age'].values
    age_Male = data[data['gender'] == 'M']['age'].values
    age_Tri = data[data['gender'] == 'T']['age'].values

    # generate plots
    fig, axs = plt.subplots(nrows=2, figsize= (15,10))
    sns.distplot(age_Female, hist = False, ax = axs[0])
    sns.distplot(age_Male, hist = False, ax = axs[0])
    fig_sns = sns.distplot(age_Tri, hist = False, ax = axs[0])

    fig_sns.set_title('Age Distribution of attacks for different gender (2013-2020)')
    fig_sns.set_ylabel('Frequency')
    fig_sns.legend(labels = ['Female', 'Male', 'Transgender'])

    # The 2nd plot is a box plot
    data['gender'] = data['gender'].replace(['F'], 'Female')
    data['gender'] = data['gender'].replace(['M'], 'Male')
    data['gender'] = data['gender'].replace(['T'], 'Transgender')
    sns.boxplot(x="gender", y="age", data=data, ax = axs[1])

    fig_sns.get_figure().savefig(outputName)
    return fig_sns


if __name__ == '__main__':
    # load csv files
    fileName = 'MergeCommon_final.csv'
    # fig_sns = age_distribution(fileName)
    # fig_sns_races = age_distribution_races(fileName)
    fig_sns_gender = age_distribution_gender(fileName)

