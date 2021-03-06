ECE143 (Fall 2020) Group 22 Project

# Analysis and Visualization of Police Violence in USA

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

matplotlib==3.2.2 <br>
numpy==1.18.5 <br>
pandas==1.0.5 <br>
geopandas==0.8.1 <br>
plotly==4.13.0 <br>
shapely==1.7.1 <br>
wordcloud==1.8.1 <br>
seaborn==0.11.0 <br>
holoviews==1.13.5 <br>
bokeh==2.2.3 <br>
xlrd==1.2.0 <br>
pywaffle==0.6.1 <br>

You can install all the packages through 
```
pip install -r requirements.txt
```

### Instructions to run script

1. Make sure you have all the dependencies installed.
2. Use the 'main' branch to access all the scripts. 
3. This repo contains all the cleaned dataset files that were used in our analysis.
4. Please run the Jupyter notebook inside the 'notebooks' folder to render and view all plots we have generated.
5. We have a few interactive plots that cannot be viewed on github. To be able to render all the plots, please clone the repo and run it on your local machine.
6. However, for your convenience, we recommend using the links below to access all of our plots.

##### Note: Please use your '@eng.ucsd.edu' email to access the files below

## Plots

#### Mapping Police Violence in USA
* [Mapping individual acts of Police Violence](https://drive.google.com/file/d/1YRSA_JK4tMGxe3LJ7ZYkXOAwuend9165/view?usp=sharing)
* [State-wise frequency of Police Violence](https://drive.google.com/file/d/1qZbK5lAs5UvA6y1uBtSmhH4k3pBj51wx/view?usp=sharing)
* [Bar plot of Police killings state-wise](https://drive.google.com/file/d/1Ml5ZY6NLOmwcRCgmFlpjprBflBQWS3iv/view?usp=sharing)

#### Trends in Police Violence
* [Number of police killings per year from 2013-2020](https://drive.google.com/file/d/1y8kWg02Nt_ab6kY0bxF4rvKwx32s6Jp6/view?usp=sharing)
* [Number of police killings versus age of victims](https://drive.google.com/file/d/1gwF8MfR2UxbzG8cdt0Igv93kbjYN--oi/view?usp=sharing)
* [Number of police killings of each gender](https://drive.google.com/file/d/1oDGnveosk2Xcl2DZTjm9A5Br-o1hFL3v/view?usp=sharing)
* [Relation between police killings and age and race of the victims](https://drive.google.com/file/d/14V1qmEVP65pcSZzn2t0-w-WV9_q1_0wv/view?usp=sharing)
* [Total number of attacks every month](https://drive.google.com/file/d/1Y8veOEQ3J6e6O7t8V5dD97RW1u2XGaeV/view?usp=sharing)

#### Racial Disparities in Police Violence
* [Bar plot showing number of police killings of different racial communities compared to their total population](https://drive.google.com/file/d/1JmvijmExzRCa2QUOTUk43jujolULuyn5/view?usp=sharing)
* [Interactive plot showing racial disparities in California, Texas and New York](https://drive.google.com/drive/folders/1OvT5mRIPAVQozuIMgJf0dgWHgfaDaOvK?usp=sharing)

#### Exploring the reasons and consequences of Police Violence
* [Word-cloud showing the weapons (if any) wielded by the victims while being attacked by the police](https://drive.google.com/file/d/1-7wlXFGNvfvXfh_EEqXQ8t9fjnJv8b08/view?usp=sharing)
* [Relation between crime rate and police violence rate](https://drive.google.com/file/d/1SYcGHQKn-8W1awSKnkuO0OqrQPfDTfns/view?usp=sharing)
* [Conviction rate of police officers](https://drive.google.com/file/d/1OyF2sy2yVhNMdUHTrHs10_a_-1pQ1Wtr/view?usp=sharing)


## Cleaned Databases
* [2013 - 2019 Killings by Police Department](https://drive.google.com/file/d/1GUNOxTpR4gk7eOgKUHz74KMwVVwgc23l/view?usp=sharing)
* [2013 - 2019 Killings by State](https://drive.google.com/file/d/1VrDPwBX59YGHt1_VcLo2_VXAS1ntFoZm/view?usp=sharing)
* [2015 - 2020 Database of Victims of Police Killings](https://drive.google.com/file/d/1tC9_Bv2mbFLoE5bvt8PLelfm3Yon-CsW/view?usp=sharing)


## [Project Presentation](https://drive.google.com/file/d/1NpZ_353YGvFJ0jOEvRwzNVKUKdauiBG3/view?usp=sharing)

## Team Members
1. Keyi Ren
2. Zhijin Liang
3. Zhaobin Huang
4. Praveen Ramani
