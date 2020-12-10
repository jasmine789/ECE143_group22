import xlrd
import pandas as pd
import numpy as np
import math
def split_excel_sheet_to_csv(filename):
    '''
    Take a excel file name as input, split each sheet in the excel file into seperate csv files.
    :param filename:
    :type filename:
    '''
    # get sheet names of the excel file
    data = xlrd.open_workbook(filename)
    print(data.sheet_names())
    # split the whole excel file into csv files
    for i in data.sheet_names():
        data_sheet = pd.read_excel(filename,i,index_col=0)
        data_sheet.to_csv(i + '.csv',encoding='utf-8')

# Washington Post
WP_shooting = pd.read_csv('../data/fatal-police-shootings-data.csv')
# Mapping police violence
MPV = pd.read_csv('../data/2013-2020 Police Killings.csv')
# print(WP_shooting.columns,len(WP_shooting.columns))
# print(MPV.columns,len(MPV.columns))
# drop different columns
WP_common = WP_shooting.drop(columns=['longitude', 'latitude', 'is_geocoding_exact'])
MPV_common = MPV.drop(columns=['Zipcode','Street Address of Incident','URL of image of victim','County',
                               'Agency responsible for death', 'ORI Agency Identifier (if available)',
                               'A brief description of the circumstances surrounding the death',
                                'Official disposition of death (justified or other)',
                                'Criminal Charges?','Link to news article or photo of official document','MPV ID', 'Fatal Encounters ID', 'Unarmed/Did Not Have an Actual Weapon',
                               'Off-Duty Killing?','Geography (via Trulia methodology based on zipcode population density: http://jedkolko.com/wp-content/uploads/2015/05/full-ZCTA-urban-suburban-rural-classification.xlsx )'])
# print(WP_common.columns)
# print(MPV_common.columns)
# rename common labels
MPV_common = MPV_common.rename(columns={"Victim's name": "name", "Victim's age": "age","Victim's gender":"gender","Victim's race":"race","Date of Incident (month/day/year)":"date",
                           "City":"city", "State":"state","Cause of death":"manner_of_death",'Symptoms of mental illness?':'signs_of_mental_illness', 'Alleged Weapon (Source: WaPo and Review of Cases Not Included in WaPo Database)':'armed',
                           'Alleged Threat Level (Source: WaPo)':'threat_level','Fleeing (Source: WaPo)':'flee','Body Camera (Source: WaPo)':'body_camera', 'WaPo ID (If included in WaPo database)':'id'
                           })
# print(WP_common.columns)
# print(MPV_common.columns)
# merge those
Merge_common = pd.merge(WP_common,MPV_common,how="outer",on=['id'])
# Merge_common.to_csv('Merge_common.csv')
# print(Merge_common.gender_x.unique())
# print(Merge_common.gender_y.unique())
# print(Merge_common.race_x.unique())
# print(Merge_common.race_y.unique())
# print(Merge_common.manner_of_death_x.unique())
# print(Merge_common.manner_of_death_y.unique())
# print(Merge_common.signs_of_mental_illness_x.unique())
# print(Merge_common.signs_of_mental_illness_y.unique())
# print(Merge_common.flee_x.unique())
# print(Merge_common.flee_y.unique())
# print(Merge_common.threat_level_x.unique())
# print(Merge_common.threat_level_y.unique())
# split_excel_sheet_to_csv('MPVDatasetDownload.xlsx')


# merge common columns
def merge_common_columns(fname1,fname2,droplist1,droplist2,renamedict1,renamedict2,oncol):
    '''
    This function merge 2 csv files based on their column names, fname1 and fname 2
    are names of 2 files, droplist1 and droplist2 are lists of columns to drop, renamelist1
    and renamelist2 are columns to be rename, on is the column name to join on
    :param fname1: file1 name
    :param fname2: file2 name
    :param clist1: columns to drop in file1
    :param clist2: columns to drop in file2
    :param renamelist1: columns to rename in file1
    :param renamelist2: columns to rename in file2
    :param on: column names to join on
    '''
    # Washington Post
    f1 = pd.read_csv(fname1)
    # Mapping police violence
    f2= pd.read_csv(fname2)
    # drop different columns and rename common columns
    f1 = f1.drop(columns = droplist1).rename(columns = renamedict1)
    f2 = f2.drop(columns = droplist2).rename(columns = renamedict2)
    # merge those
    return pd.merge(f1,f2,how="outer",on=oncol)

WP_shooting = 'fatal-police-shootings-data.csv'
MPV = '2013-2020 Police Killings.csv'
droplist1 = ['longitude', 'latitude', 'is_geocoding_exact']
droplist2 = ['Zipcode','Street Address of Incident','URL of image of victim','County',
            'Agency responsible for death', 'ORI Agency Identifier (if available)',
            'A brief description of the circumstances surrounding the death',
            'Official disposition of death (justified or other)',
            'Criminal Charges?','Link to news article or photo of official document','MPV ID', 'Fatal Encounters ID',
             'Unarmed/Did Not Have an Actual Weapon','Off-Duty Killing?',
             'Geography (via Trulia methodology based on zipcode population density: http://jedkolko.com/wp-content/uploads/2015/05/full-ZCTA-urban-suburban-rural-classification.xlsx )'
            ]
renamedict1 = dict()
renamedict2 = {"Victim's name": "name", "Victim's age": "age","Victim's gender":"gender",
               "Victim's race":"race","Date of Incident (month/day/year)":"date","City":"city",
               "State":"state","Cause of death":"manner_of_death",'Symptoms of mental illness?':'signs_of_mental_illness',
               'Alleged Weapon (Source: WaPo and Review of Cases Not Included in WaPo Database)':'armed',
               'Alleged Threat Level (Source: WaPo)':'threat_level','Fleeing (Source: WaPo)':'flee',
               'Body Camera (Source: WaPo)':'body_camera', 'WaPo ID (If included in WaPo database)':'id'
               }
on = ['id']
Merge_common = merge_common_columns('fatal-police-shootings-data.csv','2013-2020 Police Killings.csv',droplist1,droplist2,renamedict1,renamedict2,on)

# Merge_common.to_csv('Merge_common.csv')
Merge_Array = Merge_common.to_numpy()

# uniform data format
# gender
# print(Merge_common.gender_x.unique())
# print(Merge_common.gender_y.unique())
Merge_Array[Merge_Array=='Male']='M'
Merge_Array[Merge_Array=='Male ']='M'
Merge_Array[Merge_Array=='Female']='F'
Merge_Array[Merge_Array=='Unknown']=np.nan
Merge_Array[Merge_Array=='Transgender']='T'

# race
# print(Merge_common.race_x.unique())
# print(Merge_common.race_y.unique())
Merge_Array[Merge_Array=='White']='W'
Merge_Array[Merge_Array=='Black']='B'
Merge_Array[Merge_Array=='Asian']='A'
Merge_Array[Merge_Array=='Hispanic']='H'
Merge_Array[Merge_Array=='Native American']='N'
Merge_Array[Merge_Array=='Pacific Islander']='O'
Merge_Array[Merge_Array=='Unknown race']=np.nan

#sign of mental illness
# print(Merge_common.signs_of_mental_illness_x.unique())
# print(Merge_common.signs_of_mental_illness_y.unique())
Merge_Array[Merge_Array=='Yes']=True
Merge_Array[Merge_Array=='Drug or alcohol use']=False
Merge_Array[Merge_Array=='No']=False
Merge_Array[Merge_Array=='Unknown']=np.nan
Merge_Array[Merge_Array=='unknown']=np.nan
Merge_Array[Merge_Array=='Unknown ']=np.nan
Merge_Array[Merge_Array=='Unkown']=np.nan

# flee
# print(Merge_common.flee_x.unique())
# print(Merge_common.flee_y.unique())
Merge_Array[Merge_Array=='other']='Other'
Merge_Array[Merge_Array=='foot']='Foot'
Merge_Array[Merge_Array=='not fleeing']='Not fleeing'
Merge_Array[Merge_Array=='car']='Car'
Merge_Array[Merge_Array=='Not Fleeing']='Not fleeing'

# manner_of_death
Merge_Array[Merge_Array=='Gunshot']='shot'
Merge_Array[Merge_Array=='Gunshot, Taser']='shot and Tasered'

# threat_level
# print(Merge_common.threat_level_x.unique())
# print(Merge_common.threat_level_y.unique())
Merge_Array[Merge_Array=='Other']='other'

# age
# print(Merge_common.age_x.unique())
# print(Merge_common.age_y.unique())
age_x = Merge_Array[:,5]
age_y = Merge_Array[:,15]
for i in range(len(age_x)):
    if type(age_x[i]) == float and not math.isnan(age_x[i]):
        age_x[i] = int(age_x[i])
for i in range(len(age_y)):
    if type(age_y[i]) == str:
        if age_y[i].isdigit():
            age_y[i] = int(age_y[i])
        else:
            age_y[i] = np.nan

#Add items that are not in WP but in MPV to the end

rows, cols = Merge_Array.shape
for i in range(rows):
    if pd.isnull(Merge_Array[i,2]):
        Merge_Array[i,1:14] = Merge_Array[i,[14,18,21,23,15,16,17,19,20,22,24,25,26]]
columns = ['id', 'name', 'date', 'manner_of_death', 'armed', 'age', 'gender',
       'race', 'city', 'state', 'signs_of_mental_illness', 'threat_level',
       'flee', 'body_camera']

Final_csv = pd.DataFrame(Merge_Array[:,0:14],columns=columns)
Final_csv.to_csv('MergeCommon_final.csv',encoding='utf-8')
with open('MergeCommon_final.csv') as f:
    print(f)
