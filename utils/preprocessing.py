import xlrd
import pandas as pd
import numpy as np
# # get sheet names of the excel file
# data = xlrd.open_workbook('MPVDatasetDownload.xlsx')
# print(data.sheet_names())
# # split the whole excel file into csv files
# for i in data.sheet_names():
#     data_sheet = pd.read_excel('MPVDatasetDownload.xlsx',i,index_col=0)
#     data_sheet.to_csv(i + '.csv',encoding='utf-8')

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

# print(Merge_common.armed_x.unique())
# print(Merge_common.armed_y.unique())
Merge_Array = Merge_common.to_numpy()

# # achieve same format
# gender
Merge_Array[Merge_Array=='Male']='M'
Merge_Array[Merge_Array=='Male ']='M'
Merge_Array[Merge_Array=='Female']='F'
Merge_Array[Merge_Array=='Unknown']=np.nan
Merge_Array[Merge_Array=='Transgender']='T'

# race
Merge_Array[Merge_Array=='White']='W'
Merge_Array[Merge_Array=='Black']='B'
Merge_Array[Merge_Array=='Asian']='A'
Merge_Array[Merge_Array=='Hispanic']='H'
Merge_Array[Merge_Array=='Native American']='N'
Merge_Array[Merge_Array=='Pacific Islander']='O'
Merge_Array[Merge_Array=='Unknown race']=np.nan

#sign of mental illness
Merge_Array[Merge_Array=='Yes']=True
Merge_Array[Merge_Array=='Drug or alcohol use']=False
Merge_Array[Merge_Array=='No']=False
Merge_Array[Merge_Array=='Unknown']=np.nan
Merge_Array[Merge_Array=='unknown']=np.nan
Merge_Array[Merge_Array=='Unknown ']=np.nan
Merge_Array[Merge_Array=='Unkown']=np.nan

# flee
Merge_Array[Merge_Array=='other']='Other'
Merge_Array[Merge_Array=='foot']='Foot'
Merge_Array[Merge_Array=='not fleeing']='Not fleeing'
Merge_Array[Merge_Array=='car']='Car'
Merge_Array[Merge_Array=='Not Fleeing']='Not fleeing'

# manner_of_death
Merge_Array[Merge_Array=='Gunshot']='shot'
Merge_Array[Merge_Array=='Gunshot, Taser']='shot and Tasered'

# threat_level
Merge_Array[Merge_Array=='Other']='other'

# print(Merge_Array.shape)

# add items that are not in WP but in MPV
rows, cols = Merge_Array.shape
for i in range(rows):
    if pd.isnull(Merge_Array[i,2]):
        Merge_Array[i,1:14] = Merge_Array[i,[14,18,21,23,15,16,17,19,20,22,24,25,26]]
columns = ['id', 'name', 'date', 'manner_of_death', 'armed', 'age', 'gender',
       'race', 'city', 'state', 'signs_of_mental_illness', 'threat_level',
       'flee', 'body_camera']
Final_csv = pd.DataFrame(Merge_Array[:,0:14],columns=columns)
Final_csv.to_csv('MergeCommon_final.csv')
