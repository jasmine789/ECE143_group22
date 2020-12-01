# In this file, I write two pie charts to show racial disparities in a few selected states,
# including percentage of population and percentage of police killings
# The dataset used in this file is '2013-2019 Killings by State.csv'

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def racial_disparity_states_population(fileName, outputName = 'racial_disparity_states_population', statesList = ['California', 'New York', 'Texas']):
    data, populationInfo_pd, _ = extractInfo(fileName, statesList)
    fig = px.sunburst(populationInfo_pd, path = ['State', 'race'], values = 'population',
                      title = 'Percentage of Population')

    fig.write_html(outputName+'.html')  # store in html format
    fig.show()

def racial_disparity_states_killed(fileName, outputName = 'racial_disparity_states_killed', statesList = ['California', 'New York', 'Texas']):
    data, _, killedInfo_pd = extractInfo(fileName, statesList)
    fig = px.sunburst(killedInfo_pd, path=['State', 'race'], values='people_killed',
                      title='Percentage of police killings')

    fig.write_html(outputName+'.html')  # store in html format
    fig.show()


def extractInfo(fileName, statesList):
    """
    This function is to load dataset and extract information of the selected states
    :param fileName: A given file in csv format, it should contains a 'State' column
    :param statesList: a list of provided states that need to be analyzed
    :return: a pd.DataFrame contains only selected states information, population information and killed information
    """

    assert isinstance(statesList, list)
    assert len(statesList) > 0

    wholeData = pd.read_csv(fileName, engine='python')

    # only contains information of the selected states
    data = wholeData.loc[wholeData['State'].isin(statesList)]
    populationInfo = processPopulationInfo(data, statesList)
    killedInfo = processsKilledInfo(data, statesList)

    return data, populationInfo, killedInfo

def processPopulationInfo(data, statesList):
    """
    Re-format the population information into : 'State|race|population' format
    :param data: pd.DataFrame containing given state information
    :param statesList: a list of interested states
    :return: a pd.DataFrame in a given format
    """
    raceList = ['Black', 'Hispanic', 'Native American', 'Asian', 'Pacific Islander', 'White', 'Other']

    # Re-format the dataset as : State|race|population
    populationInfo = pd.DataFrame()
    raceList_inPD = []
    stateList_inPD = []
    populationList_in_PD = []

    for state in statesList:
        curStatePD = data.loc[data['State'] == state]

        for race in raceList:
            raceList_inPD.append(race)
            stateList_inPD.append(state)

            # query starts
            populationQuery = race + ' Population'
            popVal = curStatePD[populationQuery].values.item()
            populationList_in_PD.append(popVal)

    populationInfo['State'] = stateList_inPD
    populationInfo['race'] = raceList_inPD
    populationInfo['population'] = populationList_in_PD
    return populationInfo


def processsKilledInfo(data, statesList):
    """
    Re-format the killed people information into: 'State|race|people_killed' format
    :param data: pd.DataFrame containing given state information
    :param statesList: a list of interested states
    :return: a pd.DataFrame in a given format
    """
    raceList = ['Black', 'Hispanic', 'Native American', 'Asian', 'Pacific Islander', 'White', 'Unknown']

    # Re-format the dataset as : State|race|population|
    killedInfo = pd.DataFrame()
    raceList_inPD = []
    stateList_inPD = []
    killedList_in_PD = []

    for state in statesList:
        curStatePD = data.loc[data['State'] == state]
        for race in raceList:
            raceList_inPD.append(race)
            stateList_inPD.append(state)

            # query starts
            if race == 'Pacific Islander':
                killedQuery = '# ' + race + 's killed'
            elif race == 'Unknown':
                killedQuery = '# ' + race + ' Race people killed'
            else:
                killedQuery = '# ' + race + ' people killed'

            killedVal = curStatePD[killedQuery].values.item()
            killedList_in_PD.append(killedVal)

    killedInfo['State'] = stateList_inPD
    killedInfo['race'] = raceList_inPD
    killedInfo['people_killed'] = killedList_in_PD

    return killedInfo


if __name__ == '__main__':
    fileName = '2013-2019 Killings by State.csv'
    racial_disparity_states_population(fileName)
    racial_disparity_states_killed(fileName)


