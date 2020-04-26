import dash
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objs as go
from sklearn import preprocessing
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression

app = dash.Dash()
df = pd.read_csv('full_data.csv')

countries_options = sorted([dict(label=country, value=country) for country in set([location for location in df.location])], key=lambda k: k['label']) 
xAxis_options = [dict(label='Day', value='Day'),
                 dict(label='Date', value='Date')]

yAxis_function = [dict(label='None',value='None'),
                  dict(label='Log 10',value='Log 10'),
                  dict(label='% Change',value='% Change')
                 ]

yAxis_parameter = [dict(label='New Cases',value='New Cases'),
                  dict(label='New Deaths',value='New Deaths'),
                  dict(label='Total Cases',value='Total Cases'),
                  dict(label='Total Deaths',value='Total Deaths')
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
                style=dict(width='33%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='LSD_start', options=LSD_start)],
                style=dict(width='10%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='LSD_end', options=LSD_end)],
                style=dict(width='10%', display='inline-block')
        ),

        html.Div([dcc.Dropdown(id='predictForThisManyDays', options=predictForThisManyDays)],
                style=dict(width='5%', display='inline-block')
        ),

        
        dcc.Dropdown(id='countries_picker', options=countries_options, multi=True, value=['India']),
        dcc.Graph(id='graph')
], style={ "height" : "70%", "border":"1px solid blue"}
)




@app.callback(Output(component_id='graph',component_property='figure'),
                [Input(component_id='xAxis', component_property='value'),
                Input(component_id='yAxisFunction', component_property='value'),
                Input(component_id='yAxisParameter', component_property='value'),
                Input(component_id='countries_picker', component_property='value'),
                Input(component_id='LSD_start', component_property='value'),
                Input(component_id='LSD_end', component_property='value'),
                Input(component_id='predictForThisManyDays', component_property='value')])
def update_figure(x_axis, y_axis_function, y_axis_parameter, countries, LSD_start, LSD_end, extendPredictionToThisManyDays):
        data = []
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
                        xTitle = 'Number of days since the 1st reported case in the country'
                elif x_axis == 'Date':
                        xGraph = dates
                        xGraphPredicted = [ x.strftime('%Y-%m-%d') for x in pd.date_range(start=xGraph[-1], periods=extendPredictionToThisManyDays+1) ]
                        xGraphPredicted = xGraph + xGraphPredicted[1:]
                        xTitle = 'Dates'

                for parameter in y_axis_parameter:
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
                                yName = country+", log10(New Cases)"
                        elif parameter == 'New Deaths' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in new_deaths_country])
                                yName = country+", log10(New Deaths)"
                        elif parameter == 'Total Cases' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in total_cases_country])
                                yName = country+", log10(Total Cases)"
                        elif parameter == 'Total Deaths' and y_axis_function == 'Log 10':
                                yGraph = pd.Series([np.log10(a) for a in total_deaths_country])
                                yName = country+", log10(Total Deaths)"

                        elif parameter == 'New Cases' and y_axis_function == '% Change':
                                yGraph = pd.Series(new_cases_country).pct_change()*100
                                yName = country+", % change in New Cases"
                        elif parameter == 'New Deaths' and y_axis_function == '% Change':
                                yGraph = pd.Series(new_deaths_country).pct_change()*100
                                yName = country+", % change in New Cases"
                        elif parameter == 'Total Cases' and y_axis_function == '% Change':
                                yGraph = pd.Series(total_cases_country).pct_change()*100
                                yName = country+", % change in New Cases"
                        elif parameter == 'Total Deaths' and y_axis_function == '% Change':
                                yGraph = pd.Series(total_deaths_country).pct_change()*100
                                yName = country+", % change in New Cases"
                
                        data.append(go.Scatter(x=xGraph, y=yGraph, mode='markers+lines', name=yName))

                regr = LinearRegression()

                if (LSD_start == None) or (LSD_start not in xGraph):
                        LSD_start = xGraph[-1]
                
                if (LSD_end == None) or (LSD_end not in xGraph):
                        LSD_end = xGraph[-1]

                # print(xGraph)
                

                le = preprocessing.LabelEncoder()
                le2 = preprocessing.LabelEncoder()

                le.fit(xGraph)
                le2.fit(xGraphPredicted)

                dataToTrainOn = np.array(le.transform(xGraph[xGraph.index(LSD_start):xGraph.index(LSD_end)+1])).reshape((-1,1))
                print(xGraph[xGraph.index(LSD_start):xGraph.index(LSD_end)+1])
                regr.fit(dataToTrainOn, pd.Series(yGraph[xGraph.index(LSD_start):xGraph.index(LSD_end)+1].dropna()))

                # slope = (init_y - final_y) / (init_day - final_day)
                predictedLSR = regr.predict(dataToTrainOn)
                slope = (predictedLSR[0] - predictedLSR[-1]) / (le.transform([LSD_start]) - le.transform([LSD_end]))
                print("slope=",slope)
                if math.isnan(slope) == False:
                        minimisedLSRTrace = go.Scatter(x=xGraph[xGraph.index(LSD_start):xGraph.index(LSD_end)+1], y=predictedLSR, mode='lines', name=country+", LSR({:.6f})".format(slope[0]))
                        data.append(minimisedLSRTrace)
                        if extendPredictionToThisManyDays > 0:
                                dataToPredictOn = np.array(le2.transform(xGraphPredicted[xGraph.index(LSD_end)+1:xGraph.index(LSD_end)+1+extendPredictionToThisManyDays])).reshape((-1,1))
                                minimisedLSRTraceExtended = go.Scatter(x=xGraphPredicted[xGraph.index(LSD_end)+1:xGraph.index(LSD_end)+1+extendPredictionToThisManyDays], y=regr.predict(dataToPredictOn), mode='markers+lines', name=country+", LSR Extended")
                                # print(xGraphPredicted[xGraph.index(LSD_end)+1:xGraph.index(LSD_end)+1+extendPredictionToThisManyDays])
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
        




@app.callback(Output(component_id='LSD_start',component_property='options'),
                [Input(component_id='graph',component_property='figure')])
def update_LSDStart(fig):
        unique = []
        for trace in fig['data']:
                unique = trace['x'] + unique

        LSD_start = sorted([dict(label=point, value=point) for point in set(unique)], key=lambda k: k['label'])
        return LSD_start


@app.callback(Output(component_id='LSD_end',component_property='options'),
                [Input(component_id='graph',component_property='figure')])
def update_LSDEnd(fig):
        unique = []
        for trace in fig['data']:
                unique = trace['x'] + unique

        LSD_end = sorted([dict(label=point, value=point) for point in set(unique)], key=lambda k: k['label'])
        return LSD_end




# @app.callback(Output(component_id='LSD_end',component_property='value'),
#                 [Input(component_id='xAxis',component_property='value')])
# def update_LSDEnd(fig):
#         return None

# @app.callback(Output(component_id='LSD_start',component_property='value'),
#                 [Input(component_id='xAxis',component_property='value')])
# def update_LSDEnd(fig):
#         return None

if __name__ == '__main__':
        app.run_server(host="0.0.0.0")