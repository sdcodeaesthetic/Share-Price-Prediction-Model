import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0,1))
df_nse = pd.read_csv("INFY.csv")
df_nse["Date"] = pd.to_datetime(df_nse.Date, format = "%Y-%m-%d")
df_nse.index = df_nse['Date']

#Creating Dataframe
data = df_nse.sort_index(ascending = True, axis = 0)
new_data = pd.DataFrame(index = range(0, len(df_nse)), columns = ['Date', 'Close'])
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#Setting Index
new_data.index = new_data.Date
new_data.drop('Date', axis = 1, inplace = True)

#Creating Train and Test Sets
dataset = new_data.values
train = dataset[0:3500,:]
valid = dataset[3500:,:]

#Converting Dataset into x_train and y_train
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Fitting the LSTM Network
#model = Sequential()
#model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#model.add(LSTM(units = 50))
#model.add(Dense(1))
#model.compile(loss = 'mean_squared_error', optimizer = 'adam')
#model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2)
model=load_model("infymodel.h5")

#Predicting 246 Values, Using Past 60 Datas from the Train Data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs  = scaler.transform(inputs)
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
train = new_data[:3500]
valid = new_data[3500:]
valid['Predictions'] = closing_price


df = pd.read_csv("NIFTY50.csv")
app.layout = html.Div([
    html.H1("Share Price Analysis Dashboard", style = {"textAlign": "center"}),
    dcc.Tabs(id = "tabs", children = [
        dcc.Tab(label = 'INFY Stock Data', children = [
			html.Div([
				html.H2("Actual Closing Price", style = {"textAlign": "center"}),
				dcc.Graph(
					id = "Actual Data",
					figure = {
						"data":[
							go.Scatter(
								x = train.index,
								y = valid["Close"],
								mode = 'markers'
							)

						],
						"layout":go.Layout(
							title = 'Scatter Plot',
							xaxis = {'title':'Date'},
							yaxis = {'title':'Closing Rate'}
						)
					}

				),
				html.H2("Predicted Closing Price", style = {"textAlign": "center"}),
				dcc.Graph(
					id = "Predicted Data",
					figure = {
						"data":[
							go.Scatter(
								x = valid.index,
								y = valid["Predictions"],
								mode = 'markers'
							)

						],
						"layout":go.Layout(
							title='Scatter Plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		
        ]),
        dcc.Tab(label='NIFTY50 Companies Stock Data', children = [
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style = {'textAlign': 'center'}),  
                dcc.Dropdown(id = 'my-dropdown',
                             options = [{'label': 'ASIANPAINT', 'value': 'ASIANPAINT'},
                                        {'label': 'BAJAJ-AUTO','value': 'BAJAJ-AUTO'}, 
                                        {'label': 'BAJAJFINSV', 'value': 'BAJAJFINSV'}, 
                                        {'label': 'BPCL','value': 'BPCL'},
                                        {'label': 'BRITANNIA','value': 'BRITANNIA'},
                                        {'label': 'CIPLA','value': 'CIPLA'},
                                        {'label': 'HDFC','value': 'HDFC'},
                                        {'label': 'NESTLEIND','value': 'NESTLEIND'},
                                        {'label': 'RELIANCE','value': 'RELIANCE'},
                                        {'label': 'SBIN','value': 'SBIN'},
                                        {'label': 'TCS','value': 'TCS'}], 
                             multi = True, value = ['FB'],
                             style = {"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id = 'highlow'),
                html.H1("Stock's Market Volume", style = {'textAlign': 'center'}),       
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'ASIANPAINT', 'value': 'ASIANPAINT'},
                                      {'label': 'BAJAJ-AUTO','value': 'BAJAJ-AUTO'}, 
                                      {'label': 'BAJAJFINSV', 'value': 'BAJAJFINSV'}, 
                                      {'label': 'BPCL','value': 'BPCL'},
                                      {'label': 'BRITANNIA','value': 'BRITANNIA'},
                                      {'label': 'CIPLA','value': 'CIPLA'},
                                      {'label': 'HDFC','value': 'HDFC'},
                                      {'label': 'NESTLEIND','value': 'NESTLEIND'},
                                      {'label': 'RELIANCE','value': 'RELIANCE'},
                                      {'label': 'SBIN','value': 'SBIN'},
                                      {'label': 'TCS','value': 'TCS'}], 
                             multi = True, value = ['FB'],
                             style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id = 'volume')
            ], className = "container"),
        ])
    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"ASIANPAINT": "ASIANPAINT", "BAJAJ_AUTO": "BAJAJ_AUTO", "BAJAJFINSV": "BAJAJFINSV", "BPCL": "BPCL", "BRITANNIA": "BRITANNIA", "CIPLA": "CIPLA", "HDFC": "HDFC", "NESTLEIND": "NESTLEIND", "RELIANCE": "RELIANCE", "SBIN": "SBIN", "TCS": "TCS",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x = df[df["Symbol"] == stock]["Date"],
                     y = df[df["Symbol"] == stock]["High"],
                     mode = 'lines', opacity = 0.7, 
                     name = f'High {dropdown[stock]}',textposition = 'bottom center'))
        trace2.append(
          go.Scatter(x = df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["Low"],
                     mode = 'lines', opacity = 0.6,
                     name = f'Low {dropdown[stock]}',textposition = 'bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway = ["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'], height=600,
               title = f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis = {"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis = {"title":"Price (INR)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"ASIANPAINT": "ASIANPAINT", "BAJAJ_AUTO": "BAJAJ_AUTO", "BAJAJFINSV": "BAJAJFINSV", "BPCL": "BPCL", "BRITANNIA": "BRITANNIA", "CIPLA": "CIPLA", "HDFC": "HDFC", "NESTLEIND": "NESTLEIND", "RELIANCE": "RELIANCE", "SBIN": "SBIN", "TCS": "TCS",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                     y=df[df["Symbol"] == stock]["VWAP"],
                     mode='lines', opacity=0.7,
                     name=f'VWAP {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway = ["#5E0DAC", '#FF4F00', '#375CB1', 
                                              '#FF7400', '#FFF400', '#FF0056'], height = 600,
            title = f"VWAP for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis = {"title":"Date",
                     'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                     'rangeslider': {'visible': True}, 'type': 'date'},
            yaxis = {"title":"Transactions Volume"})}
    return figure


if __name__=='__main__':
	app.run_server(debug = True)
