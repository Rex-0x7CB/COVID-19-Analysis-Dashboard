import dash
import math
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
from sklearn import preprocessing
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

app = dash.Dash()
df = pd.read_csv('covid-19-data\public\data\ecdc\\full_data.csv')

countries_options = sorted([dict(label=country, value=country) for country in set([location for location in df.location])], key=lambda k: k['label']) 
xAxis_options = [dict(label='Day', value='Day'),
                 dict(label='Date', value='Date')]

correctedError_options = [dict(label='Prediction Without Corrected Error', value=True),
                 dict(label="Prediction With Corrected Error", value=False)]


yAxis_function = [dict(label='None',value='None'),
                  dict(label='Log 10',value='Log 10'),
                  dict(label='% Change',value='% Change')
                 ]

yAxis_parameter = [dict(label='New Cases',value='New Cases'),
                  dict(label='New Deaths',value='New Deaths'),
                  dict(label='Total Cases',value='Total Cases'),
                  dict(label='Total Deaths',value='Total Deaths')
                  ]

algorithm_options = [dict(label='Least Squared Residual', value ='Least Squared Residual'),
                     dict(label='Bayesian Regression', value='Bayesian Regression')
                ]

LSD_start = []
LSD_end = []

predictForThisManyDays = sorted([dict(label=i, value=i) for i in [i for i in range(0,32)]], key=lambda k: k['label'])

app.layout = html.Div([

        html.Div([dcc.Dropdown(id='xAxis', options=xAxis_options, value='Day')],
                style=dict(width='10%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='yAxisFunction', options=yAxis_function, value='None')],
                style=dict(width='10%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='yAxisParameter', options=yAxis_parameter, multi=True, value=['New Cases'])],
                style=dict(width='30%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='algorithm', multi=True, options=algorithm_options, value=['Least Squared Residual'])],
                style=dict(width='29%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='LSD_start', options=LSD_start)],
                style=dict(width='8%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='LSD_end', options=LSD_end)],
                style=dict(width='8%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='predictForThisManyDays', options=predictForThisManyDays)],
                style=dict(width='5%', display='inline-block')
        ),
        
        html.Div([dcc.Dropdown(id='countries_picker', options=countries_options, multi=True, value=['India'])],
                style=dict(width='70%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='correctedError', options=correctedError_options, value=False)],
                style=dict(width='18%', display='inline-block')
        ),


        html.Div([html.Button(id='predict_total', n_clicks=0, children='Predict')],
                style=dict(width='5%', display='inline-block')
        ),

        dcc.Graph(id='graph-main'),

        html.Div([dcc.Graph(id='predicted-daily')],
                style=dict(width='50%', display='inline-block')
        ),

        html.Div([dcc.Graph(id='predicted-total')],
                style=dict(width='50%', display='inline-block')
        )

            
]
)


def calculatePredicted(country, y, init_date, traceData):
        df = pd.read_csv('covid-19-data\public\data\ecdc\\full_data.csv')
        df = df[ df['location'] == country ]
        if traceData == 'Total Deaths':
                col = 'total_deaths'
        elif traceData == 'Total Cases':
                col = 'total_cases'

        if type(init_date) == type('str'):
                df = df[ df['date'] == init_date ]
                init_infec = df.iloc[0][col]

        if (type(init_date) == type(1)):
                df = df[ df['total_cases'] > 0 ]
                init_infec = df.iloc[init_date][col]

        dailyPredictedCases = []
        totalPredictedCases = []

        for perc in y:
                new_case = init_infec * (perc/100)
                if new_case < 0:
                        new_case = 0
                init_infec = init_infec + new_case
                dailyPredictedCases.append(new_case)
                totalPredictedCases.append(int(init_infec))

        return (dailyPredictedCases, totalPredictedCases)



@app.callback(Output(component_id='graph-main',component_property='figure'),
                [Input(component_id='xAxis', component_property='value'),
                Input(component_id='yAxisFunction', component_property='value'),
                Input(component_id='yAxisParameter', component_property='value'),
                Input(component_id='countries_picker', component_property='value'),
                Input(component_id='LSD_start', component_property='value'),
                Input(component_id='LSD_end', component_property='value'),
                Input(component_id='predictForThisManyDays', component_property='value'),
                Input(component_id='algorithm', component_property='value')])
def update_figure(x_axis, y_axis_function, y_axis_parameter, countries, LSD_start, LSD_end, extendPredictionToThisManyDays, algorithm):
        data = []
        xTitle = ''
        print(countries)
        print(x_axis)
        print(y_axis_function)
        print(y_axis_parameter)

        for country in countries:
                print(country)
                found = False
                dates = []
                new_cases_country = []
                new_deaths_country = []
                total_cases_country = []
                total_deaths_country = []
                for row in df.iterrows():
                        if x_axis == 'Day':
                                condition = row[1].location == country and (row[1].new_cases > 0 or found == True)
                        elif x_axis == 'Date':
                                condition = row[1].location == country
                        if condition:
                                dates.append(row[1].date)
                                new_cases_country.append(row[1].new_cases)
                                new_deaths_country.append(row[1].new_deaths)
                                total_cases_country.append(row[1].total_cases)
                                total_deaths_country.append(row[1].total_deaths)
                                found = True
                
                if extendPredictionToThisManyDays == None:
                        extendPredictionToThisManyDays = 0

                if x_axis == 'Day':
                        xGraph = list(range(0,len(total_cases_country)))
                        xGraphPredicted = list(range(0,len(total_cases_country)+extendPredictionToThisManyDays))
                        xTitle = 'Number of days since the 1st reported case in a country'
                elif x_axis == 'Date':
                        xGraph = dates
                        xGraphPredicted = [ x.strftime('%Y-%m-%d') for x in pd.date_range(start=xGraph[-1], periods=extendPredictionToThisManyDays+1) ]
                        xGraphPredicted = xGraph + xGraphPredicted[1:]
                        xTitle = 'Dates'

                for parameter in y_axis_parameter:
                        if y_axis_function == None:
                                y_axis_function = 'None'
                        print(parameter)
                        if parameter == 'New Cases' and y_axis_function == 'None':
                                yGraph = pd.Series(new_cases_country)
                                yName = country+", {}".format(parameter)
                        elif parameter == 'New Deaths' and y_axis_function == 'None':
                                yGraph = pd.Series(new_deaths_country)
                                yName = country+", {}".format(parameter)
                        elif parameter == 'Total Cases' and y_axis_function == 'None':
                                yGraph = pd.Series(total_cases_country)
                                yName = country+", {}".format(parameter)
                        elif parameter == 'Total Deaths' and y_axis_function == 'None':
                                yGraph = pd.Series(total_deaths_country)
                                yName = country+", {}".format(parameter)

                        elif parameter == 'New Cases' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in new_cases_country])
                                yName = country+", log10({})".format(parameter)
                        elif parameter == 'New Deaths' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in new_deaths_country])
                                yName = country+", log10({})".format(parameter)
                        elif parameter == 'Total Cases' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in total_cases_country])
                                yName = country+", log10({})".format(parameter)
                        elif parameter == 'Total Deaths' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in total_deaths_country])
                                yName = country+", log10({})".format(parameter)

                        elif parameter == 'New Cases' and y_axis_function == '% Change':
                                yGraph = pd.Series(new_cases_country).pct_change()*100
                                yName = country+", % change in {}".format(parameter)
                        elif parameter == 'New Deaths' and y_axis_function == '% Change':
                                yGraph = pd.Series(new_deaths_country).pct_change()*100
                                yName = country+", % change in {}".format(parameter)
                        elif parameter == 'Total Cases' and y_axis_function == '% Change':
                                yGraph = pd.Series(total_cases_country).pct_change()*100
                                yName = country+", % change in {}".format(parameter)
                        elif parameter == 'Total Deaths' and y_axis_function == '% Change':
                                yGraph = pd.Series(total_deaths_country).pct_change()*100
                                yName = country+", % change in {}".format(parameter)
                
                        data.append(go.Scatter(x=xGraph, y=yGraph, mode='markers+lines', name=yName))
                

                        for algo in algorithm:
                                print(algo)
                                if algo == 'Least Squared Residual':
                                        regr = LinearRegression()
                                        algo_name = 'LSR'
                                elif algo == 'Bayesian Regression':
                                        regr = BayesianRidge()
                                        algo_name = 'Linear Bayes'

                                if (LSD_start == None) or (LSD_start not in xGraph):
                                        LSD_start = xGraph[-1]
                                
                                if (LSD_end == None) or (LSD_end not in xGraph):
                                        LSD_end = xGraph[-1]
                                

                                le = preprocessing.LabelEncoder()

                                le.fit(xGraphPredicted)

                                dataToTrainOn = np.array(le.transform(xGraphPredicted[xGraphPredicted.index(LSD_start):xGraphPredicted.index(LSD_end)+1])).reshape((-1,1))
                                regr.fit(dataToTrainOn, pd.Series(yGraph[xGraphPredicted.index(LSD_start):xGraphPredicted.index(LSD_end)+1].dropna()))
                                
                                # print("#####################################")
                                # print(xGraphPredicted[xGraphPredicted.index(LSD_start):xGraphPredicted.index(LSD_end)+1])
                                # print(yGraph[xGraphPredicted.index(LSD_start):xGraphPredicted.index(LSD_end)+1].dropna())
                                # print("#####################################")

                                # slope = (init_y - final_y) / (init_day - final_day)
                                predictedLSR = regr.predict(dataToTrainOn)
                                slope = (predictedLSR[0] - predictedLSR[-1]) / (le.transform([LSD_start]) - le.transform([LSD_end]))
                                if math.isnan(slope) == False:
                                        minimisedLSRTrace = go.Scatter(x=xGraphPredicted[xGraphPredicted.index(LSD_start):xGraphPredicted.index(LSD_end)+1], y=predictedLSR, mode='lines', name=country+", "+algo_name+"({:.6f})".format(slope[0]))
                                        data.append(minimisedLSRTrace)
                                        if extendPredictionToThisManyDays > 0:
                                                dataToPredictOn = np.array(le.transform(xGraphPredicted[xGraphPredicted.index(LSD_end)+1:xGraphPredicted.index(LSD_end)+1+extendPredictionToThisManyDays])).reshape((-1,1))
                                                minimisedLSRTraceExtended = go.Scatter(x=xGraphPredicted[xGraphPredicted.index(LSD_end)+1:xGraphPredicted.index(LSD_end)+1+extendPredictionToThisManyDays], y=regr.predict(dataToPredictOn), mode='markers+lines', name=country+", "+algo_name+" Extended")
                                                data.append(minimisedLSRTraceExtended)


        layout = go.Layout(title='COVID-19 Transmission Analysis',
                                xaxis=dict(title=xTitle),
                                yaxis=dict(title='Count'),
                                height=650,
                                showlegend=True,
                                hovermode='x'
                                )
        
        fig = go.Figure(data=data, layout=layout)
        # fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
        
        # fig.add_trace(go.Scatter(x=xGraph, y=[15000]*len(total_cases_country), mode='lines', line=dict(color='red'), name='15,000 Cases'))
        # fig.add_trace(go.Scatter(x=xGraph, y=[20000]*len(total_cases_country), mode='lines', line=dict(color='red'), name='20,000 Cases'))
        # fig.add_trace(go.Scatter(x=xGraph, y=[25000]*len(total_cases_country), mode='lines', line=dict(color='red'), name='25,000 Cases'))
        # fig.add_trace(go.Scatter(x=xGraph, y=[30000]*len(total_cases_country), mode='lines', line=dict(color='red'), name='30,000 Cases'))

        return fig


def calculateAverageError(graph,countryName):
        country = ""
        init_date = ""
        traceData = ""
        temp = 0

        if graph != None:
                for trace in graph['data']:
                        if 'LSR(' in trace['name']:
                                calculatedTraceX = trace['x']
                                calculatedTraceY = trace['y']
                                calBegPoint = trace['x'][0]
                                calEndPoint = trace['x'][-1]
                                temp +=1

                        if '% change' in trace['name']:
                                realTraceX = trace['x']
                                realTraceY = trace['y']
                                temp +=1

                        if 'LSR Extended' in trace['name']:
                                country = trace['name'].replace(", LSR Extended", "")
                                extendedTraceY = trace['y']
                                temp +=1

                        if temp % 3 == 0 :
                                calBegPointIndex = realTraceX.index(calBegPoint)
                                calEndPointIndex = realTraceX.index(calEndPoint)
                                realTraceX = realTraceX[calBegPointIndex:calEndPointIndex+1]
                                realTraceY = realTraceY[calBegPointIndex:calEndPointIndex+1]

                                errorTraceY = [ y-x for y,x in zip(calculatedTraceY,realTraceY) ]
                                errorTraceY = [ y/x if x!=0 else 0 for y,x in zip(errorTraceY,realTraceY) ]
                                errorTraceY = [ x*100 for x in errorTraceY ]

                                averageError = sum(errorTraceY)/len(errorTraceY)

                                if countryName == country:
                                        print(country)
                                        # print("\tcalculated TraceX=", calculatedTraceX)
                                        # print("\tcalculated TraceY=", calculatedTraceY)
                                        # print("\trealTraceX=", realTraceX)
                                        # print("\trealTraceY=", realTraceY)
                                        # print("\terrorTraceY=", errorTraceY)
                                        print("\taverageError=", averageError)
                                        print("\n")
                                        return averageError
                                

        

@app.callback(Output(component_id='predicted-daily',component_property='figure'),
                [Input(component_id='predict_total', component_property='n_clicks')],
                [State(component_id='graph-main',component_property='figure')])
def update_predictedDaily(n_clicks, graph):
        country = ""
        init_date = ""
        traceData = ""
        temp = 0
        data = []

        # print(graph)
        if graph != None:
                for trace in graph['data']:
                        if 'LSR(' in trace['name']:
                                init_date = trace['x'][-1]
                                temp +=1

                        if '% change' in trace['name']:
                                traceData = trace['name'].replace(", % change in ", "")
                                print("traceData", traceData)
                                temp +=1

                        if 'LSR Extended' in trace['name']:
                                country = trace['name'].replace(", LSR Extended", "")
                                traceData = traceData.replace(country, "")
                                extendedTraceY = trace['y']
                                temp +=1

                        if temp % 3 == 0 :
                                averageError = calculateAverageError(graph, country)
                                (predictedDailyCases, predictedTotalCases) = calculatePredicted(country, extendedTraceY, init_date, traceData)
                                predictedDailyCases = [ x-(averageError*x)/100 for x in predictedDailyCases ]

                                data.append(go.Bar(x=trace['x'], y=predictedDailyCases, name=country+", Daily New "+traceData.replace("Total ","")+"(Predicted)"))
                                


        if type(init_date) == type('str'):
                xTitle = 'Dates'

        if (type(init_date) == type(1)):
                xTitle = 'Days since 1st discovery'
        
        layout = go.Layout(title='COVID-19 Predictions',
                                xaxis=dict(title=xTitle),
                                yaxis=dict(title='Count'), showlegend=True, hovermode='x')
        fig = go.Figure(data=data, layout=layout)
        return fig
        

@app.callback(Output(component_id='predicted-total',component_property='figure'),
                [Input(component_id='predict_total', component_property='n_clicks')],
                [State(component_id='graph-main',component_property='figure'), State(component_id='correctedError',component_property='value')])
def update_predictedTotal(n_clicks, graph, correctedErrorOption):
        print("Current State Of Graph")
        country = ""
        traceData = ""
        xTitle = ""
        data = []
        initDateAxis = []
        temp = 0
        df = pd.read_csv('covid-19-data\public\data\ecdc\\full_data.csv')
        if graph != None:
                for trace in graph['data']:
                        if 'LSR(' in trace['name']:
                                temp += 1
                                initDate = trace['x'][-1]

                        if '% change' in trace['name']:
                                temp += 1
                                traceData = trace['name'].replace(", % change in ", "")
                                initDateAxis = trace['x']

                        if 'LSR Extended' in trace['name']:
                                temp += 1
                                country = trace['name'].replace(", LSR Extended", "")
                                traceData = traceData.replace(country, "")
                                extendedTraceY = trace['y']
                                extendedTraceX = trace['x']
                                (predictedDailyCases, predictedTotalCases) = calculatePredicted(country, extendedTraceY , initDate, traceData)

                                if traceData == 'Total Deaths':
                                        col = 'total_deaths'
                                elif traceData == 'Total Cases':
                                        col = 'total_cases'
                        
                        if temp % 3 == 0:
                                totalPredictedCases = []
                                averageError = calculateAverageError(graph, country)
                                dfCountry = df[ df['location'] == country ]

                                if type(initDate) == type('str'):
                                        tempDf = dfCountry[ dfCountry['date'] == initDate ]
                                        init_infec = tempDf.iloc[0][col]

                                        totalCases = dfCountry[col].values.tolist()
                                        data.append(go.Scatter(x=initDateAxis, y=totalCases, mode='markers+lines', name=country+", "+traceData+"(Actual)"))
                                        xTitle = 'Dates'

                                if (type(initDate) == type(1)):
                                        tempDf = dfCountry[ dfCountry['total_cases'] > 0 ]
                                        init_infec = tempDf.iloc[initDate][col]

                                        dfCountry = dfCountry[ dfCountry[col] > 0 ]
                                        totalCases = dfCountry[col].values.tolist()
                                        data.append(go.Scatter(x=initDateAxis, y=totalCases, mode='markers+lines', name=country+", "+traceData+"(Actual)"))
                                        xTitle = 'Days since 1st discovery'

                                count = 0
                                predictedDailyCases = [ x-(averageError*x)/100 for x in predictedDailyCases ]
                        
                                for perc in extendedTraceY:
                                        init_infec = init_infec + predictedDailyCases[count]
                                        totalPredictedCases.append(int(init_infec))
                                        count+=1
                                if correctedErrorOption:
                                        data.append(go.Scatter(x=extendedTraceX, y=predictedTotalCases, mode='markers+lines', name=country+", "+traceData+"(Predicted)"))
                                data.append(go.Scatter(x=extendedTraceX, y=totalPredictedCases, mode='markers+lines', name=country+", "+traceData+"(Predicted EC)"))

        
        
        layout = go.Layout(title='COVID-19 Predictions',
                                xaxis=dict(title=xTitle),
                                yaxis=dict(title="Count"), showlegend=True, hovermode='x')
        fig = go.Figure(data=data, layout=layout)
        return fig


@app.callback(Output(component_id='LSD_start',component_property='options'),
                [Input(component_id='graph-main',component_property='figure')])
def update_LSDStart(fig):
        unique = []
        for trace in fig['data']:
                unique = trace['x'] + unique

        LSD_start = sorted([dict(label=point, value=point) for point in set(unique)], key=lambda k: k['label'])
        return LSD_start


@app.callback(Output(component_id='LSD_end',component_property='options'),
                [Input(component_id='graph-main',component_property='figure')])
def update_LSDEnd(fig):
        unique = []
        for trace in fig['data']:
                unique = trace['x'] + unique

        LSD_end = sorted([dict(label=point, value=point) for point in set(unique)], key=lambda k: k['label'])
        return LSD_end

if __name__ == '__main__':
        app.run_server(host="0.0.0.0")