ECE143 (Fall 2020) Group 22 Project

# Visualizing and Modeling Police Violence in USA

<img src="/images/police_violence.png" height="250" width="1550" alt="SDMTS">

## Description
Police Violence is a serious social issue and has attracted a lot of attention from the media off late. In this project, we aim to analyze trends in police violence using two expansive and continually updated datasets:

* [Washington Post Police Shootings database](https://github.com/washingtonpost/data-police-shootings):<br/>
This dataset contains a csv file contains id, name, date, manner of death, armed, age, gender, race, city, state, signs of mental illness, threat_level, flee, body_camera,         latitude and longitude of the shooting location and is_geocoding_exact of police shooting since 2015.
* [Mapping Police Violence Database](https://mappingpoliceviolence.org/):<br/>
This dataset is an excel file contains name, age, gender, race, date, street, city, state, zip code, county, related Agency and code,cause of death, brief description, official disposition of death, criminal charges, symptoms of mental illness, armed, threat level, armed, weapon, fleeing, body camera,Wapo/MPV/Fatal Encounters id, geography, off-duty killing, image link and news article link of police violence since 2013.

This project tries to analyze:

* When and where acts of police violence have occured
* Correlation between crime rate and violence rate of each state
* Correlation between age of victims and violence against them
* Amount of charges filed against police officers for committing unnecessary acts of violence
* Racial disparities in acts of deadly violence at the central and state level

## Code Organization

### Requirements and Dependencies

1. matplotlib
2. numpy
3. pandas
4. geopandas
5. plotly
6. shapely
7. wordcloud
8. datetime
9. seaborn
10. holoviews

### Instructions to run script

_To be completed_

1. Make sure you have all the dependencies installed.
2. Use the 'main' branch to access all the scripts. 
3. This repo contains all the cleaned dataset files that were used in our analysis.
4. We have a few interactive plots that cannot be viewed on github. To be able to render all the plots, please clone the repo and run it on your local machine.
5. However, for your convenience, we recommend using the links below to access all of our plots.

## Plots

_Add links here_

### Mapping Police Violence in USA
1. [Mapping individual acts of Police Violence](https://github.com/jasmine789/ECE143_group22/blob/main/images/occurancy_per_city.png)
2. [State-wise frequency of Police Violence](https://github.com/jasmine789/ECE143_group22/blob/main/images/killings_per_state_map.png)
3. [Bar plot of Police killings state-wise](https://github.com/jasmine789/ECE143_group22/blob/main/images/Killing_per_state_bar.png)

## Cleaned Databases
* [2013 - 2019 Killings by Police Department](https://drive.google.com/file/d/1GUNOxTpR4gk7eOgKUHz74KMwVVwgc23l/view?usp=sharing)
* [2013 - 2019 Killings by State](https://drive.google.com/file/d/1VrDPwBX59YGHt1_VcLo2_VXAS1ntFoZm/view?usp=sharing)
* [2015 - 2020 Database of Victims of Police Killings](https://drive.google.com/file/d/1tC9_Bv2mbFLoE5bvt8PLelfm3Yon-CsW/view?usp=sharing)

Note: Use '@eng.ucsd.edu' e-mail ID to access these files

## Team Members
1. Keyi Ren
2. Zhijin Liang
3. Zhaobin Huang
4. Praveen Ramani
