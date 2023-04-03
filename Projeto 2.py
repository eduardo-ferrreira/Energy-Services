# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:51:59 2023

@author: eduar
"""
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data and prepare it

df_power_2019 = pd.read_csv('testData_2019_Central.csv') #Open test file
df_power_2019['Date'] = pd.to_datetime(df_power_2019['Date']) # Convert Date into datetime type
df_power_2019['Hour']=df_power_2019['Date'].dt.hour #Column with respective hour
df_power_2019['Power-1']=df_power_2019['Central (kWh)'].shift(1) # Column with previous hour consumption
df_power_2019 = df_power_2019.dropna() # Errases previously NaN created
df_power_2019.drop(columns=['temp_C', 'HR','pres_mbar', 'solarRad_W/m2', 
                            'rain_mm/h', 'rain_day'], inplace=True) #remove non-features
df_power_2019=df_power_2019.iloc[:, [1, 0, 5, 4, 2, 3]] # Change the position of the columns so that Y=column 0 and X all the remaining columns

df = df_power_2019


#Create X (features) and Y (Power) for plotting

X = df.iloc[:, 2:6]
fig = px.line(df, x = "Date", y = df.columns[2:6])

Y = df['Central (kWh)'].values


#Load and run models

with open('BT_model.pkl','rb') as file:
    BT_model=pickle.load(file)

Y_pred_BT = BT_model.predict(X)

#Evaluate errors
MAE_BT=metrics.mean_absolute_error(Y,Y_pred_BT) 
MSE_BT=metrics.mean_squared_error(Y,Y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(Y,Y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(Y)


#Load RF model
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)

Y_pred_RF = RF_model.predict(X)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(Y, Y_pred_RF) 
MSE_RF=metrics.mean_squared_error(Y, Y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(Y, Y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(Y)


#Load GB model
with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)

Y_pred_GB = GB_model.predict(X) 

#Evaluate errors
MAE_GB=metrics.mean_absolute_error(Y, Y_pred_GB) 
MSE_GB=metrics.mean_squared_error(Y, Y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(Y, Y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(Y)


#Load XGB model
with open('XGB_model.pkl','rb') as file:
    XGB_model=pickle.load(file)

Y_pred_XGB = XGB_model.predict(X) 

#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(Y, Y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(Y, Y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(Y, Y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(Y)


#Load NN model
with open('NN_model.pkl','rb') as file:
    NN_model=pickle.load(file)

Y_pred_NN = NN_model.predict(X) 

#Evaluate errors
MAE_NN=metrics.mean_absolute_error(Y, Y_pred_NN) 
MSE_NN=metrics.mean_squared_error(Y, Y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(Y, Y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(Y)


#Load LR model
with open('LR_model.pkl','rb') as file:
    LR_model=pickle.load(file)

Y_pred_LR = LR_model.predict(X) 

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(Y, Y_pred_LR) 
MSE_LR=metrics.mean_squared_error(Y, Y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(Y, Y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(Y)


#Load  model
with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)

Y_pred_GB = GB_model.predict(X) 

#Evaluate errors
MAE_GB=metrics.mean_absolute_error(Y, Y_pred_GB) 
MSE_GB=metrics.mean_squared_error(Y, Y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(Y, Y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(Y)


#Creating dataframe of power and model predictions

d = {'Methods': ['Bootstrapping', 'Random Forest', 'Gradient Boosting', 'Extreme Gradient Boosting',
                 'Neural Networks', 'Linear Regression'], 'MAE': [MAE_BT, MAE_RF, MAE_GB, MAE_XGB, MAE_NN, MAE_LR],
     'MSE': [MSE_BT, MSE_RF, MSE_GB, MSE_XGB, MSE_NN, MSE_LR], 'RMSE': [RMSE_BT, RMSE_RF, RMSE_GB, RMSE_XGB, RMSE_NN, RMSE_LR],
     'cvMSE': [cvRMSE_BT, cvRMSE_RF, cvRMSE_GB, cvRMSE_XGB, cvRMSE_NN, cvRMSE_LR]}
df_metrics = pd.DataFrame(data=d)

d = {'Date': df['Date'].values, 'Bootstrapping': Y_pred_BT, 'RandomForest': Y_pred_RF, 'Gradient Boosting': Y_pred_GB, 
     'Extreme Gradient Boosting' : Y_pred_XGB, 'Neural Networks': Y_pred_NN, "Linear Regression": Y_pred_LR }
df_forecast = pd.DataFrame(data = d)
df_results = pd.merge(df, df_forecast, on = 'Date')
df_results = df_results.drop(columns=['Power-1', 'Hour', 'windSpeed_m/s', 'windGust_m/s'])
df_results = df_results.iloc[:, [1, 0, 2, 3, 4,5,6,7]]

fig2 = px.line(df_results, x = df_results.columns[0], y = df_results.columns[1:7])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

import base64

image_filename = 'IST_Logo.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([
    html.Img(src=image_filename.format(encoded_image.decode()),
             style={'height':'8%', 'width':'8%', 'float':'right', 'position':'relative', 'margin-top': '-50px'}),
    html.H2('IST Energy Forecast Tool (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])



app.layout = html.Div([
    html.H2('IST Energy Forecast Tool (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig,
            ),
            
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            generate_table(df_metrics)
        ])

if __name__ == '__main__':
    app.run_server(debug=False) 