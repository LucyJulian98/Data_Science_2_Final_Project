import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import sqlite3
#import missingno as miss
from matplotlib import pyplot as plt
import seaborn as sns
#from vega_datasets import data
import plotly.express as px

### For Modeling
# For pre-processing data
from sklearn import preprocessing as pp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

# Pipeline to combine modeling elements
from sklearn.pipeline import Pipeline

### For Modeling and Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # Cross validation
from sklearn.model_selection import cross_validate # Cross validation
from sklearn.model_selection import GridSearchCV # Cross validation + param. tuning.

# Machine learning methods
from sklearn.linear_model import LinearRegression as LM
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor as DTree
from sklearn.ensemble import RandomForestRegressor as RF

### Metrics
import sklearn.metrics as m

# For model interpretation
from sklearn.inspection import (
    permutation_importance,
    partial_dependence,
    PartialDependenceDisplay,
    plot_partial_dependence
)

### Connecting to SQLite Database
conn = sqlite3.connect("Climate_Change_Project.sqlite")

df = pd.read_sql("SELECT * FROM Final_Database", con = conn)
#df.shape
df = df.rename(columns = {"Primary energy consumption (TWh)" : "Energy_Consumption_TWh", "Forest cover" : "Forest_Cover"})

df = df[df.columns.difference(["Country", "Year", "Energy_Consumption_TWh", "Literacy_Rate"])]
#miss.matrix(df)




#Add sidebar to the app
st.sidebar.markdown("### Data Science 2 Project App")
st.sidebar.markdown("This app is powered by Streamlit . It provides \
                     the option to view interesting visualizations and/or \
                     look at the regression results!")
Options = ["Home", "Visualizations", "Country Level Regression", "Electrical Vehicle Regression", "Income_Groups"]
option = st.sidebar.selectbox("Select an Option", Options)
if option == "Home" :
    st.title("Welcome to Climate Change Project App")
    #st.markdown
    st.subheader("Please Select an Option from the Sidebar to View Different Results")
if option == "Visualizations" :

    st.title("Visualizing indicators used in Regression Analysis")
    ### Comment
    st.subheader("Choose desired visualization : ")
    Viz_Options = ["Forest Area by Landmass",
                "Income Groups"]
    viz_option = st.selectbox("Choose an Option", Viz_Options)
    if viz_option == "Forest Area by Landmass" :
        ## Forest Area Increase
        data = pd.read_sql("SELECT * FROM Forest_Data", con = conn)
        data['Year'] = pd.to_datetime(data['Year'], format = '%Y')
        countries = list(set(data["Country"]))
        countries.sort()

        country_list = st.multiselect("Choose a Country", countries)
        country_plot_df = data.loc[data.Country.isin(country_list)]
        #st.write(country_plot_df)
        alt_chart = alt.Chart(country_plot_df).mark_line(color = 'green').encode(
        x = alt.X("Year", axis=alt.Axis(title='Year')),
        y = alt.Y("Forest cover", axis=alt.Axis(title='Forest Area by Percentage of Landmass'), scale = alt.Scale(domain = (0, 100))),
        color = "Country"
        ).properties(
        width = 1000,
        height = 800
        )
        st.altair_chart(alt_chart)

    if viz_option == "Income Groups" :
        data = pd.read_sql("SELECT * FROM Income_Groups", con = conn)
        data = data.dropna()
        fig = px.choropleth(data, locations="ISO_3",
                    color="Income_Group",
                    hover_name="Country",
                    width = 1000,
                    height = 800,
                    animation_frame = 'Year',
                    animation_group = 'Country')
        st.plotly_chart(fig)


if option == "Country Level Regression" :
   st.title("Results of Country Level Regression Analysis")
   predictors = ["Forest_Cover", "ag_land_index", "Corruption_Rank", "Corruption Perception Index", "NDC_Submitted", "Internet_User_Percentage", "Liberal_Democracy_Index", "Renewable_Energy_Electricity_Percentage"]
   pred_list = st.multiselect("Choose the Predictors", predictors)

   pred_list.append("GDP_US_Dollars")
   pred_list.append("Population")
   pred_list.append("H")
   pred_list.append("LM")
   pred_list.append("UM")
   pred_list.append("Annual_CO2_Emissions")

   final_df = df.filter(pred_list)
   final_df = final_df.dropna()
   ### Displaying the Data Frame
   st.subheader("Sneak Peek of the Final Dataframe : ")
   st.write(final_df.sample(20))

   X = final_df[final_df.columns.difference(["Annual_CO2_Emissions"])]
   y = final_df["Annual_CO2_Emissions"]


   y = np.log(y+1) ### Logging emissions
   X["ln_GDP_US_Dollars"] = np.log(X["GDP_US_Dollars"] + 1)
   X["ln_Population"] = np.log(X["Population"] + 1)
   X = X.drop(columns = ["GDP_US_Dollars", "Population"])
   corr_matrix = X.corr() ###  Getting the correlation matrix
   ## Plotting correlation matrix
   st.subheader("Correlation Heatmap of Feature Matrix : ")
   plt.figure(figsize=(60,40), dpi = 100) ## Setting the graph size
   fig, ax = plt.subplots()
   sns.set(font_scale = 1.5) ## Setting the font size
   sns.heatmap(corr_matrix, cmap = "YlGnBu", ax = ax)
   st.write(fig)


   ## Splitting into training and testing data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 75)

   # (1) Set the folds index to ensure comparable samples
   fold_generator = KFold(n_splits=5, shuffle=True,random_state=111)

   # (2) Specifying the preprocessing steps
   #preprocess = ColumnTransformer(transformers=[('num', pp.MinMaxScaler(), p_temp)])

   # (3) Creating the model pipeline
   pipe = Pipeline(steps=[('model',None)])

   search_space = [
    # Decision Tree with the Max Depth Param
    {'model': [DTree()],
     'model__max_depth':[2,4,10,15]}

   ]

   # (5) Putting it all together in the grid search
   search = GridSearchCV(pipe, search_space,
                          cv = fold_generator,
                          scoring='r2',
                          n_jobs=4)

   # (6) Fitting the model to the training data
   search.fit(X_train, y_train)

   ## Modeling
   st.subheader("Decision Tree Model Results : ")

   ### Get the best model
   """
   The best performing model is :
   """
   st.write(search.best_params_)
   """
   The R2 score from the model is :
   """
   st.write(search.best_score_)

   ### Evaluating performance on test set
   mod = search.best_estimator_ ## Getting the best model
   ## Mean squred error on test data
   pred_y = search.predict(X_test) ## Predicting the y values using model
   """
   The R2 score from the model for test dataset is :
   """
   st.write(m.r2_score(y_test, pred_y))

   ### Model Interpretability
   st.subheader("Variables of Importance : ")
   ## Computing permutation importance
   vi = permutation_importance(mod,X_train,y_train,n_repeats= 25)
   # Organize as a data frame
   vi_dat = pd.DataFrame(dict(variable=X_train.columns,
                           vi = vi['importances_mean'],
                           std = vi['importances_std']))

   # Generate intervals
   vi_dat['low'] = vi_dat['vi'] - 2*vi_dat['std']
   vi_dat['high'] = vi_dat['vi'] + 2*vi_dat['std']

   # But in order from most to least important
   vi_dat = vi_dat.sort_values(by="vi",ascending=False).reset_index(drop=True)

   alt_chart = alt.Chart(vi_dat).mark_bar().encode(
                x = alt.X('vi', axis=alt.Axis(title='Change in Error')),
                y = alt.Y('variable', axis=alt.Axis(title='Variable'))
   ).properties(
     width = 800,
     height = 600
   )
   st.altair_chart(alt_chart)

if option == "Income_Groups" :
   st.title("Results of Country Level Regression Analysis based on Income Groups")
   Income_levels = ['Low', 'Lower Middle', 'Upper Middle', 'High']
   income = st.selectbox("Select an Income Level", Income_levels)
   if income == 'High' :
       df = df.loc[df.H == 1]
   elif income == 'Upper Middle' :
       df = df.loc[df.UM == 1]
   elif income == 'Lower Middle' :
       df = df.loc[df.LM == 1]
   else :
       df = df.loc[(df.H == 0) & (df.UM ==0) & (df.LM == 0)]
    ### Displaying the Data Frame


   st.subheader("Sneak Peek of the Final Dataframe : ")
   st.write(df.sample(20))

   df = df.drop(columns = ["H", "LM", "UM"])
   ### Preprocessing
   final_df = df.dropna()
   X = final_df[final_df.columns.difference(["Annual_CO2_Emissions"])]
   y = final_df["Annual_CO2_Emissions"]
   y = np.log(y+1) ### Logging emissions
   #X["Energy_Consumption_TWh"] = np.log(X["Energy_Consumption_TWh"] + 1)
   X["ln_GDP_US_Dollars"] = np.log(X["GDP_US_Dollars"] + 1)
   X["ln_Population"] = np.log(X["Population"] + 1)
   X = X.drop(columns = ["GDP_US_Dollars", "Population"])
   corr_matrix = X.corr() ###  Getting the correlation matrix

   ## Plotting correlation matrix
   st.subheader("Correlation Heatmap of Feature Matrix : ")
   plt.figure(figsize=(60,40), dpi = 100) ## Setting the graph size
   fig, ax = plt.subplots()
   sns.set(font_scale = 1.5) ## Setting the font size
   sns.heatmap(corr_matrix, cmap = "YlGnBu", ax = ax)
   st.write(fig)

   ## Modeling
   st.subheader("Decision Tree Model Results : ")
   ## Splitting into training and testing data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 75)

   # (1) Set the folds index to ensure comparable samples
   fold_generator = KFold(n_splits=5, shuffle=True,random_state=111)

   # (2) Specifying the preprocessing steps
   #preprocess = ColumnTransformer(transformers=[('num', pp.MinMaxScaler(), ['Liberal_Democracy_Index', 'ln_Population', 'ag_land_index', 'Forest_Cover', 'Renewable_Energy_Electricity_Percentage', 'ln_GDP_US_Dollars'])])

   # (3) Creating the model pipeline
   pipe = Pipeline(steps=[('model',None)])

   search_space = [
    # Decision Tree with the Max Depth Param
    {'model': [DTree()],
     'model__max_depth':[2,4,10,15]}

   ]

   # (5) Putting it all together in the grid search
   search = GridSearchCV(pipe, search_space,
                          cv = fold_generator,
                          scoring='r2',
                          n_jobs=4)

   # (6) Fitting the model to the training data
   search.fit(X_train, y_train)

   ### Get the best model
   """
   The best performing model is :
   """
   st.write(search.best_params_)
   """
   The R2 score from the model is :
   """
   st.write(search.best_score_)

   ### Evaluating performance on test set
   mod = search.best_estimator_ ## Getting the best model
   ## Mean squred error on test data
   pred_y = search.predict(X_test) ## Predicting the y values using model
   """
   The R2 score from the model for test dataset is :
   """
   st.write(m.r2_score(y_test, pred_y))

   ### Model Interpretability
   st.subheader("Variables of Importance : ")
   ## Computing permutation importance
   vi = permutation_importance(mod,X_train,y_train,n_repeats= 25)
   # Organize as a data frame
   vi_dat = pd.DataFrame(dict(variable=X_train.columns,
                           vi = vi['importances_mean'],
                           std = vi['importances_std']))

   # Generate intervals
   vi_dat['low'] = vi_dat['vi'] - 2*vi_dat['std']
   vi_dat['high'] = vi_dat['vi'] + 2*vi_dat['std']

   # But in order from most to least important
   vi_dat = vi_dat.sort_values(by="vi",ascending=False).reset_index(drop=True)

   alt_chart = alt.Chart(vi_dat).mark_bar().encode(
                x = alt.X('vi', axis=alt.Axis(title='Change in Error')),
                y = alt.Y('variable', axis=alt.Axis(title='Variable'))
   ).properties(
     width = 800,
     height = 600
   )
   st.altair_chart(alt_chart)






if option == "Electrical Vehicle Regression" :
   df = pd.read_sql("SELECT * FROM Electrical_Vehicle", con = conn)
   st.title("Results of Electrical Vehicles Regression Analysis for Countries in Europe")
   ### Displaying the Data Frame
   st.subheader("Sneak Peek of the Final Dataframe : ")
   st.write(df.sample(20))

   ### Preprocessing
   final_df = df.dropna()
   X = final_df[final_df.columns.difference(["Country", "Year", "Annual_CO2_Emissions"])]
   y = final_df["Annual_CO2_Emissions"]
   y = np.log(y+1) ### Logging emissions
   X["ln_Population"] = np.log(X["Population"] + 1)
   X = X.drop(columns = ["Population"])

   corr_matrix = X.corr()
   ## Plotting correlation matrix
   st.subheader("Correlation Heatmap of Feature Matrix : ")
   plt.figure(figsize=(60,40), dpi = 100) ## Setting the graph size
   fig, ax = plt.subplots()
   sns.set(font_scale = 1.5) ## Setting the font size
   sns.heatmap(corr_matrix, cmap = "YlGnBu", ax = ax)
   st.write(fig)


   ## Splitting into training and testing data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 75)

   # (1) Set the folds index to ensure comparable samples
   fold_generator = KFold(n_splits=5, shuffle=True,random_state=111)

   # (2) Specifying the preprocessing steps
   preprocess = ColumnTransformer(transformers=[('num', pp.MinMaxScaler(), ['ln_Population', 'Renewable_Energy_Electricity_Percentage', 'battery_electric_share'])])

   # (3) Creating the model pipeline
   pipe = Pipeline(steps=[('pre_process', preprocess),
                           ('model',None)])

   search_space = [
    # Decision Tree with the Max Depth Param
    {'model': [DTree()],
     'model__max_depth':[1,2,3,5,10]}

   ]

   # (5) Putting it all together in the grid search
   search = GridSearchCV(pipe, search_space,
                          cv = fold_generator,
                          scoring='r2',
                          n_jobs=4)

   # (6) Fitting the model to the training data
   search.fit(X_train, y_train)

   ## Modeling
   st.subheader("Decision Tree Model Results : ")

   ### Get the best model
   """
   The best performing model is :
   """
   st.write(search.best_params_)
   """
   The R2 score from the model is :
   """
   st.write(search.best_score_)

   ### Evaluating performance on test set
   mod = search.best_estimator_ ## Getting the best model
   ## Mean squred error on test data
   pred_y = search.predict(X_test) ## Predicting the y values using model
   """
   The R2 score from the model for test dataset is :
   """
   st.write(m.r2_score(y_test, pred_y))

   ### Model Interpretability
   st.subheader("Variables of Importance : ")
   ## Computing permutation importance
   vi = permutation_importance(mod,X_train,y_train,n_repeats= 25)
   # Organize as a data frame
   vi_dat = pd.DataFrame(dict(variable=X_train.columns,
                           vi = vi['importances_mean'],
                           std = vi['importances_std']))

   # Generate intervals
   vi_dat['low'] = vi_dat['vi'] - 2*vi_dat['std']
   vi_dat['high'] = vi_dat['vi'] + 2*vi_dat['std']

   # But in order from most to least important
   vi_dat = vi_dat.sort_values(by="vi",ascending=False).reset_index(drop=True)

   alt_chart = alt.Chart(vi_dat).mark_bar().encode(
                x = alt.X('vi', axis=alt.Axis(title='Change in Error')),
                y = alt.Y('variable', axis=alt.Axis(title='Variable'))
   ).properties(
     width = 800,
     height = 600
   )
   st.altair_chart(alt_chart)
