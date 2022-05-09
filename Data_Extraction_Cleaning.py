### Code for extracting and cleaning data

import pandas as pd
import numpy as np
import sqlite3
import country_converter as coco
import pycountry

### For API Calls
import json
import requests
from bs4 import BeautifulSoup

conn = sqlite3.connect("Climate_Change_Project.sqlite")

def country_code_generator(dataframe) :
    """
    Function to convert country names to ISO3 codes
    """
    dataframe['Country'] = coco.convert(names = dataframe['Country'], to = 'name_short')
    dataframe = dataframe[dataframe['Country'] != 'not found']
    return dataframe


### Viewing historical emissions data

emissions_data = pd.read_csv("/Users/lucienne/Documents/Studies/Data_Science_Projects/Data_Science_2_Project/Data/historical_emissions/historical_emissions.csv")
#emissions_data.sample(10)
df = emissions_data.filter(["Country", "Gas", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011", "2010"]) \
df = df.loc[df.Gas == "CO2"]
df = df.drop(columns = ["Gas"])
df = df.melt(id_vars = "Country", value_vars = df.columns.difference(["Country"]), var_name = 'Year', value_name = "Emissions")
#df.sample(10)

energy_data = pd.read_csv("Data/Energy_Consumption.csv")
#energy_data.sample(10)
energy_df = (energy_data.filter(["Entity", "Year", "Primary energy consumption (TWh)"]) \
                        .rename(columns = {"Entity" : "Country"}))

energy_df = country_code_generator(energy_df)
energy_df = energy_df.loc[energy_df.Year>=2010]
energy_df.sample(5)

len(set(energy_df['Country']))

agriculture_data = pd.read_csv("Data/agricultural_labor_land.csv")
agriculture_df = (agriculture_data.filter(["Entity", "Year", "ag_land_index"]) \
                                  .rename(columns = {'Entity' : 'Country'}))
agriculture_df = country_code_generator(agriculture_df)
agriculture_df = agriculture_df.loc[agriculture_df.Year>=2010]
agriculture_df.sample(5)

len(set(agriculture_df['Country']))

Forest_Area_data = pd.read_csv("Data/Forest_Area_By_Landmass.csv")
Forest_Area_data.sample(10)

Forest_df = (Forest_Area_data.filter(["Entity", "Year", "Forest cover"]) \
                             .rename(columns = {"Entity" : "Country"}))
Forest_df = country_code_generator(Forest_df)
### Writing to SQL
Forest_df.loc[Forest_df.Year>=1950].to_sql("Forest_Data", con = conn, index = False)

Forest_df = Forest_df.loc[Forest_df.Year>=2010]
Forest_df.sample(5)

len(set(Forest_df['Country']))

Renewables_data = pd.read_csv("Data/share-electricity-renewables.csv")
Renewables_data.sample(10)

Renewables_df = (Renewables_data.filter(["Entity", "Year", "Renewables (% electricity)"]) \
                                .rename(columns = {"Entity" : "Country", "Renewables (% electricity)" : "Renewable_Energy_Electricity_Percentage"}))
Renewables_df = country_code_generator(Renewables_df)

Renewables_df = Renewables_df.loc[Renewables_df.Year>=2010]
Renewables_df.sample(5)

len(set(Renewables_df['Country']))

CO2_emissions_data = pd.read_csv("Data/annual-co2-emissions-per-country.csv")
CO2_emissions_data.sample(10)

CO2_df = (CO2_emissions_data.filter(["Code", "Year", "Annual CO2 emissions"]) \
                            .rename(columns = {"Code" : "Country", "Annual CO2 emissions" : "Annual_CO2_Emissions"}))
CO2_df = country_code_generator(CO2_df)

### Removing lists from the CO2_df Country columns

set(CO2_df["Country"])


CO2_df = CO2_df.loc[CO2_df.Year>=2010]
CO2_df.sample(50)




CO2_df.to_sql("Climate_Change_Project", con = conn, index = False)

Democracy_data = pd.read_csv("Data/Country_Year_V-Dem_Core_CSV_v11.1/V-Dem-CY-Core-v11.1.csv")
Democracy_data.sample(10)

Democracy_df = (Democracy_data.filter(["country_name", "year", "v2x_libdem"])
                              .rename(columns = {"country_name" : "Country", "year" : "Year", "v2x_libdem" : "Liberal_Democracy_Index"}))
Democracy_df = country_code_generator(Democracy_df)

Democracy_df = Democracy_df.loc[Democracy_df.Year>=2010]
Democracy_df.sample(5)

len(set(Democracy_df['Country']))

GDP_data = pd.read_csv("Data/GDP_Data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3358362.csv", skiprows = 4)
GDP_data.sample(10)
GDP_df = (GDP_data.filter(["Country Name", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]) \
                  .rename(columns = {"Country Name" : "Country"}))
GDP_df = GDP_df.melt(id_vars = ["Country"], value_vars = GDP_df.columns.difference(["Country"]), var_name = "Year", value_name = "GDP_US_Dollars")
GDP_df = country_code_generator(GDP_df)
GDP_df["Year"] = GDP_df["Year"].astype('int64')
GDP_df.sample(5)

len(set(GDP_df['Country']))

Pop_data = pd.read_csv("Data/Population_Data/API_SP.POP.TOTL_DS2_en_csv_v2_3358390.csv", skiprows = 4)
Pop_data.sample(10)
Pop_df = (Pop_data.filter(["Country Name", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]) \
                  .rename(columns = {"Country Name" : "Country"}))

Pop_df = Pop_df.melt(id_vars = ["Country"], value_vars = Pop_df.columns.difference(["Country"]), var_name = "Year", value_name = "Population")
Pop_df["Year"] = Pop_df["Year"].astype('int64')
Pop_df = country_code_generator(Pop_df)

Pop_df.sample(5)
len(set(Pop_df['Country']))

###
NDC_data = pd.read_csv("Data/ndc_content/ndc_content.csv")
NDC_data.loc[NDC_data.Country == "India"]

Country_list = list(set(NDC_data.Country))
n = len(Country_list)
Year_dummy = np.zeros(n)
dict = {'Country' : Country_list,
        '2010' : Year_dummy,
        '2011' : Year_dummy,
        '2012' : Year_dummy,
        '2013' : Year_dummy,
        '2014' : Year_dummy,
        '2015' : Year_dummy,
        '2016' : Year_dummy,
        '2017' : Year_dummy,
        '2018' : Year_dummy,
        '2019' : Year_dummy,
        '2020' : Year_dummy}

NDC_df = pd.DataFrame(data = dict)
temp_1 = NDC_data.filter(["Country", "Indicator ID", "Value"])

temp_1 = (temp_1.pivot(index = ["Country"], columns = "Indicator ID", values = "Value")
          .reset_index()
          .filter(["Country", "pa_ratified", "pa_ratified_date"]))
temp_1["Ratified_Year"] = temp_1["pa_ratified_date"].str[-4:]
NDC_df = NDC_df.merge(temp_1.filter(["Country", "Ratified_Year"]), on = "Country", how = "left")


for i in range(2010, 2021) :
    NDC_df[str(i)] = np.where(NDC_df.Ratified_Year > str(i), 0, 1)

NDC_df = NDC_df.drop(columns = ["Ratified_Year"])
NDC_df = NDC_df.melt(id_vars = ["Country"], value_vars = NDC_df.columns.difference(["Country"]), var_name = "Year", value_name = "NDC_Submitted")
NDC_df = NDC_df.astype({'Year' :'int64'})
NDC_df.sample(10)

Income_Groups_data = pd.read_excel("Data/OGHIST.xls", sheet_name = "Country Analytical History", skiprows = 5)
#Income_Groups_data.head(10)

Income_Groups_df = (Income_Groups_data.drop(Income_Groups_data.index[:5]) \
                    .filter(["Data for calendar year :", 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]) \
                    .rename(columns = {"Data for calendar year :" : "Country"}))
Income_Groups_df = Income_Groups_df.melt(id_vars = ["Country"], value_vars = Income_Groups_df.columns.difference(["Country"]), var_name = "Year", value_name = "Income_Group")
### Creating a duplicate for Country variable
Income_Groups_df['ISO_3'] = Income_Groups_df['Country']

Income_Groups_df = country_code_generator(Income_Groups_df)

Income_Groups_df.to_sql("Income_Groups", con = conn, index = False)
Income_dummies = pd.get_dummies(Income_Groups_df["Income_Group"]) ## Dropping one column to avoid multicollinearity


Income_Groups_df = Income_Groups_df.join(Income_dummies, how = "left").drop(columns=["Income_Group", "L", '..'])
Income_Groups_df.head(10)


Internet_Percentage_df = pd.read_csv('Data/Internet_Users_by_Percentage.csv', skiprows = 4)
Internet_Percentage_df = (Internet_Percentage_df.filter(["Country Name", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]) \
                          .rename(columns = {"Country Name" : "Country"}))
Internet_Percentage_df = Internet_Percentage_df.melt(id_vars = ["Country"], value_vars = Internet_Percentage_df.columns.difference(["Country"]), var_name = "Year", value_name = "Internet_User_Percentage")
Internet_Percentage_df = country_code_generator(Internet_Percentage_df)
Internet_Percentage_df = Internet_Percentage_df.astype({'Year' :'int64'})
len(set(Internet_Percentage_df['Country']))
Internet_Percentage_df.head(10)

Literacy_Rate_df = pd.read_csv("Data/Literacy_Rate_by_Percentage.csv", skiprows = 4)
Literacy_Rate_df = (Literacy_Rate_df.filter(["Country Name", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]) \
                          .rename(columns = {"Country Name" : "Country"}))
Literacy_Rate_df = Literacy_Rate_df.melt(id_vars = ["Country"], value_vars = Literacy_Rate_df.columns.difference(["Country"]), var_name = "Year", value_name = "Literacy_Rate")
Literacy_Rate_df = country_code_generator(Literacy_Rate_df)
Literacy_Rate_df = Literacy_Rate_df.astype({'Year' :'int64'})
len(set(Literacy_Rate_df['Country']))
Literacy_Rate_df.head(10)

Corruption_df = pd.read_csv("Data/Corruption_Index.csv")
Corruption_df = (Corruption_df.filter(["Country Name", "Indicator", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]) \
                          .rename(columns = {"Country Name" : "Country"}))
Corruption_df = Corruption_df.melt(id_vars = ["Country", "Indicator"], value_vars = Corruption_df.columns.difference(["Country", "Indicator"]), var_name = "Year", value_name = "Corruption Index")
Corruption_df = (Corruption_df.pivot(index = ['Country', 'Year'], columns = 'Indicator', values = 'Corruption Index')
                 .reset_index()
                 .rename(columns = {'CPI Score' : 'Corruption Perception Index', 'Rank' : 'Corruption_Rank'}))

Corruption_df = country_code_generator(Corruption_df)
Corruption_df = Corruption_df.astype({'Year' :'int64'})
len(set(Corruption_df['Country']))

Corruption_df.head(10)

### Constructing the final dataframe and writing to SQLite Database
final_df = (pd.merge(energy_df, agriculture_df, how = "left", on = ["Country", "Year"])
            .merge(Forest_df, how = "left", on = ["Country", "Year"])
            .merge(Renewables_df, how = "left", on = ["Country", "Year"])
            .merge(Democracy_df, how = "left", on = ["Country", "Year"])
            .merge(GDP_df, how = "left", on = ["Country", "Year"])
            .merge(Pop_df, how = "left", on = ["Country", "Year"])
            .merge(Income_Groups_df, how = "left", on = ["Country", "Year"])
            .merge(Internet_Percentage_df, how = "left", on = ["Country", "Year"])
            .merge(Literacy_Rate_df, how = "left", on = ["Country", "Year"])
            .merge(Corruption_df, how = "left", on = ["Country", "Year"])
            .merge(NDC_df, how = "left", on = ["Country", "Year"])
            .merge(CO2_df, how = "left", on = ["Country", "Year"]))


final_df.sample(10)



final_df.to_sql("Final_Database", con= conn, index = False)

Electric_Vehicle_data = pd.read_csv("/Users/lucienne/Documents/Studies/Data_Science_Projects/Data_Science_2_Project/Data/share-vehicle-electric.csv")
Electric_Vehicle_data.sample(10)

Electric_Vehicle_df = (Electric_Vehicle_data.filter(["Entity", "Year", "battery_electric_share"]) \
                                            .rename(columns = {"Entity" : "Country"}))

Electric_Vehicle_df = Electric_Vehicle_df.loc[Electric_Vehicle_df.Year >=2010]

Electric_Vehicle_df.sample(10)

final_electric_df = (pd.merge(Electric_Vehicle_df, Income_Groups_df, how = 'left', on = ['Country', 'Year']) \
                     .merge(Renewables_df, how = 'left', on = ['Country', 'Year']) \
                     .merge(Pop_df, how = 'left', on = ['Country', 'Year']) \
                     .merge(CO2_df, how = 'left', on = ['Country', 'Year']))

final_electric_df.sample(10)

##3 Writing to SQL database
final_electric_df.to_sql("Electrical_Vehicle", con = conn, index = False)

conn.close()
