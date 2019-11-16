import dash
from dash.dependencies import Output, Event, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import sqlite3
import pandas as pd
import time
import numpy as np
# from google.cloud import translate
#
# translate_client = translate.Client()
# def translate(tweet):
# 	return str(translate_client.translate(tweet,target_language='en')['translatedText'])

global number_clicks
global topic_input
#global dfm
#global add_topic
#add_topic = [["USA"]]
topic_input = [{'label': 'USA', 'value': 'USA'},
            {'label': 'Trump', 'value': 'Trump'},
            {'label': 'Obama', 'value': 'Obama'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Games', 'value': 'games'},
            {'label': 'UK', 'value': 'UK'},
            {'label': 'France', 'value': 'France'},
            {'label': 'Hello', 'value': 'hello'},
            {'label': 'Japan', 'value': 'Japan'}]

dft = pd.read_csv('test.csv')

dt = pd.read_csv("trends_data.csv",encoding = "ISO-8859-1")
dt['trends']= dt['trends'].apply(eval)

df3 = pd.DataFrame(dt['trends'].values.tolist(), columns=['A','B','C','D','E','F','G','H','I','J'])

for col in dft.columns:
    dft[col] = dft[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

dt['text'] = dft['state'] + '<br>' +\
    'TRENDS:'+'<br>'+df3['A']+'<br>'+\
    df3['B']+'<br>'+df3['C']+'<br>'+df3['D']+\
    '<br>'+df3['E']+'<br>'+df3['F']+'<br>'\
    +df3['G']+'<br>'+df3['H']+'<br>'+df3['I']+'<br>'+df3['J']
        #print(dt)

data = [dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = dft['code'],
            z = dft['activity'].astype(float),
            locationmode = 'USA-states',
            text = dt['text'],
            marker = dict(
                line = dict (
                    color = 'rgb(255,255,255)',
                    width = 1
                ) ),
            colorbar = dict(
                title = "Tweet Activities")
            ) ]

layout = dict(
        title ="TYPE IN THE KEYWORD OF YOUR CHOICE TO START." + "<br>"+ " This process could take a while...",
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
            height = 600,
            )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#54278F'
}
app = dash.Dash(__name__)
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


app_map = html.Div(children=[
    html.H1(children='Streaming-Tweets-Visualization',
            style = {'textAlign': 'center',
                    'color':colors['text']}),

    html.Div(children='''

        We can easily get latest news and world events from trends panel in Twitter app. But how cool is that if we could check what people are talking about within a specific country, or state? In this project, we are doing a data visualized map which uses live stream data from Twitter API and shows Twitter trends in different locations. We will also make the map such that users will be able to choose a keyword from top trends and see what the activity of the trend in any state. Meanwhile, we also want to conduct sentiments analysis throughout the visualization. Our project will be useful in real life situations. For example, when a company release a new product, the twitter map will show how people in different state are reflecting on their new product. With sentiment analysis it will also show how people are thinking of the new product. Thus the company could use those information to perform different marketing strategy with a specific aim at each state.


    ''',
        style = {
            'textAlign': 'left',
            'color': colors['text'],
        }),
    html.Br(),
    html.Div(
    dcc.Input(id='input', type='text', value=''),
        style={
            'textAlign': 'center',
        }),
    html.Div(
    html.Button(id='submit', type='submit', children='submit'),
            style={
            'textAlign': 'center',
        }),
    html.Div(children = [dcc.Graph(
        id='example-graph',
        figure = {
            'data':data,
            'layout':layout
        }
    )]),
    	html.Div(children=[

		dcc.Graph(
			id = 'selected-data',
			figure = dict(

				data = [go.Bar(x=dft['state'], y=dft['activity'])],
				layout = dict(
					paper_bgcolor = '#F4F4F8',
					plot_bgcolor = '#F4F4F8',
					height = 600
				)
			),
			# animate = True
		)
	]),

], className="six columns")


app_sent = html.Div(children=[html.H3('Real-Time Twitter Sentiments Analysis by Topics'),
	html.Br(),

	html.H6('submit any topic below or select from the dropdown menu',
  			style={
            'color': "blue"}),
    html.Div([
  			dcc.Input(id='term', type='text', className="six columns"),
  			html.A("SUBMIT", href='/', className="six columns")
  		], className="row"),
  		html.Div(id='topic-output'),

    dcc.Dropdown(
        id='sentiment_term',
        options=topic_input,
        #value = topic_input[-1]["label"],
        multi = True),
        html.Br(),
        html.H5('Real-Time Sentiments Trend by Topics',
  			style={
            'textAlign': 'center',
            'color': "blue"}),
        dcc.Graph(id='rt-graph', animate=False),
        html.H6('drag above bar to select timeframe',
        style={
            'textAlign': 'center'}),
        html.Br(),
        html.H5('Histogram for Topics Statistics',
  			style={
            'textAlign': 'center',
            'color': "blue"}),
        dcc.Graph(id='rt-hist', animate=False),
        dcc.Interval(id='graph-update',interval=5*1000),
        html.Button('Pause/Continue', id='button'),
        html.H5(id='button-clicks'),
        html.Br(),
        #html.H2('Real-Time Twitter Sentiments by Topics'),
        html.Div([
  			html.Div(html.H5('Recent ten tweets contain the topic:',
  			style={
            'textAlign': 'center',
            'color': "blue"}), className="six columns"),
  			html.Div(html.H5('The ten tweets with Translations(currently disabled):',
  			style={
            'textAlign': 'center',
            'color': "blue"}), className="six columns"),
  		], className="row"),
  		html.Br(),
  		html.Div([
  			html.Div(id='live-tweet', className="six columns"),
  			html.Div(id='live-tweet-trans', className="six columns")
  		], className="row")], className="six columns")

app.layout = html.Div([app_map, app_sent], className="row")
conn1 = sqlite3.connect('twitter_old.db')
    #c = conn.cursor()
c1 = conn1.cursor()
temp1 = 'NA'
#temp1 = "DELETE FROM sentiment WHERE location LIKE 'NA', "
c1.execute("DELETE FROM sentiment WHERE location LIKE 'NA' ",)

@app.callback(
    Output('topic-output', 'children'),
    [Input('term', 'n_submit'), Input('term', 'n_blur')],
    [State('term', 'value')])
def input_topic(ns1, nb1, input1):
	#add_topic.append([input1])
	#print(add_topic)
	new_term = {'label': input1, 'value': input1}
	if (new_term != '') and (new_term not in topic_input):
		topic_input.append(new_term)
	return input1
#print(topic_input)

number_clicks = 0
@app.callback(
    Output('button-clicks', 'children'),
    [Input('button', 'n_clicks')])
def clicks(n_clicks):
	global number_clicks
	number_clicks = int(n_clicks)
	#print(number_clicks)
	return number_clicks


#print(number_clicks)
@app.callback(Output('rt-graph', 'figure'),
			  [Input(component_id='sentiment_term', component_property='value')],
			  events=[Event('graph-update', 'interval')])
def update_graph_scatter(sentiment_term):
	if number_clicks % 2 == 0:
		l = []
		Y_l = []

		for x in sentiment_term:
			try:
				conn = sqlite3.connect('twitter_new.db')
				c = conn.cursor()
				df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + x + '%',))
				df.sort_values('unix', inplace=True)
				df['sentiment_rolling'] = df['sentiment'].rolling(int(len(df)/5)).mean()

				df['date'] = pd.to_datetime(df['unix'],unit='ms')
				df.set_index('date', inplace=True)

				df = df.resample('1s').mean()
				df.dropna(inplace=True)
				X = df.index
				Y = df.sentiment_rolling

				vars()[x]=plotly.graph_objs.Scatter(
						x=X,
						y=Y,
						name= x,
						mode= 'lines+markers'
						)
			except:
				pass

		for i in sentiment_term:
			l.append(eval(i))
			Y_l.append(eval(i)['y'])
		if len(sentiment_term) !=1:
			Y_l = np.concatenate(Y_l).ravel().tolist()
		#print(Y_l)
		#print(min(X))
		#print(X[len(X)-1])
		return {'data': l,'layout' : go.Layout(height=650,xaxis=dict(range=[X[len(X)-50],X[len(X)-1]],
				rangeselector=dict(
				buttons=list([
					dict(count=10,
						 label='10s',
						 step='second',
						 stepmode='backward'),
					dict(count=60,
						 label='1m',
						 step='second',
						 stepmode='backward'),
					dict(count=10,
						label='10m',
						step='minute',
						stepmode='backward'),
					dict(count=1,
						label='1h',
						step='hour',
						stepmode='backward'),
					dict(step='all')
				])
			),
				rangeslider=dict(visible = True)),
				yaxis=dict(range=[-1,1]),
				title='Topics: {} <br> (1 means absolutely positive, -1 means absolutely negative.) '.format(sentiment_term))}

@app.callback(Output('rt-hist', 'figure'),
			  [Input(component_id='sentiment_term', component_property='value')],
			  events=[Event('graph-update', 'interval')])
def update_graph_scatter(sentiment_term):
	if number_clicks % 2 == 0:
		l = []
		Y_l = []

		for x in sentiment_term:
			try:
				conn = sqlite3.connect('twitter_new.db')
				c = conn.cursor()
				df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + x + '%',))
				df.sort_values('unix', inplace=True)
				#df['sentiment_rolling'] = df['sentiment'].rolling(int(len(df)/5)).mean()

				df['date'] = pd.to_datetime(df['unix'],unit='ms')
				df.set_index('date', inplace=True)

				df = df.resample('1s').mean()
				df.dropna(inplace=True)
				#X = df.index[-1]
				X = [x]
				Y = [int(len(df.index))]

				vars()[x]={
						'x': X,
						'y': Y,
						'name': x,
						'type': 'bar'
						}
			except:
				pass

		for i in sentiment_term:
			l.append(eval(i))

		return {'data': l,'layout' : {'yaxis':dict(range=[0,1000]),'title': 'Number of Tweets by Topics'}}


@app.callback(Output('live-tweet', 'children'),
			  [Input(component_id='sentiment_term', component_property='value')],
			  events=[Event('graph-update', 'interval')])
def update_tweet(sentiment_term):
	global df_2
	print(number_clicks)
	if number_clicks % 2 == 0:
		l = []
		for x in sentiment_term:
			try:
				conn = sqlite3.connect('twitter_new.db')
				c = conn.cursor()
				df_2 = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + x[0] + '%',))
				df_2.sort_values('unix', inplace=True)
				#df_2['tweet_trans'] = df_2['tweet'].apply(translate)

			except Exception as e:
				with open('errors.txt','a') as f:
					f.write(str(e))
					f.write('\n')
		return 'Tweet: {}'.format(df_2['tweet'][:10])
	else:
		return 'Tweet: {}'.format(df_2['tweet'][:10])

@app.callback(Output('live-tweet-trans', 'children'),
			  [Input(component_id='sentiment_term', component_property='value')],
			  events=[Event('graph-update', 'interval')])
def update_tweet(sentiment_term):
	global df_3
	if number_clicks % 2 == 0:
		l = []
		Y_l = []
		for x in sentiment_term:
			try:
				conn = sqlite3.connect('twitter_new.db')
				c = conn.cursor()
				df_3 = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + x[0] + '%',))
				df_3.sort_values('unix', inplace=True)
				#df_2['tweet_trans'] = df_2['tweet'].apply(translate)

			except Exception as e:
				with open('errors.txt','a') as f:
					f.write(str(e))
					f.write('\n')
		#pd.options.display.max_colwidth = 500
		return 'Tweet: {}'.format(df_3['tweet'][:10])
	else:
		return 'Tweet: {}'.format(df_3['tweet'][:10])


@app.callback(
    Output('example-graph', 'figure'),
    [],[State('input', 'value')], [Event('submit', 'click')])
def update_graph(state):

    states = dict()
    with open('states.csv','r') as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        words = line.split(",")
        states[words[0]] = [words[1],0,int(words[2])]
# import sqlite3
    #print (22, time.time())
    conn = sqlite3.connect('twitter_old.db')
    #c = conn.cursor()
    temp = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC", conn ,params=('%' + state + '%',))
    #print my_id
    for index,row in temp.iterrows():
        temp = row['location'].upper()
        #print(temp)
        # if "LOS ANGELES" in temp:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!")
        if ", AL" in temp or "Alabama".upper() in temp:
            states['AL'][1] += 1
        if ", AK" in temp or "Alaska".upper() in temp:
            states['AK'][1] += 1
        if ", AR" in temp or "Arkansas".upper() in temp:
            states['AR'][1] += 1
        if ", AZ" in temp or "Arizona".upper() in temp:
            states['AZ'][1] += 1
        if ", CA" in temp or "California".upper() in temp:
            #print("****************************")
            states['CA'][1] += 1
        if ", CO" in temp or "Colorado".upper() in temp:
            states['CO'][1] += 1
        if ", CT" in temp or "Connecticut".upper() in temp:
            states['CT'][1] += 1
        if ", DE" in temp or "Delaware".upper() in temp:
            states['DE'][1] += 1
        if ", FL" in temp or "Florida".upper() in temp:
            states['FL'][1] += 1
        if ", GA" in temp or "Georgia".upper() in temp:
            states['GA'][1] += 1
        if ", HI" in temp or "Hawaii".upper() in temp:
            states['HI'][1] += 1
        if ", IA" in temp or "Iowa".upper() in temp:
            states['IA'][1] += 1
        if ", ID" in temp or "Idaho".upper() in temp:
            states['ID'][1] += 1
        if ", IL" in temp or "Illinois".upper() in temp:
            states['IL'][1] += 1
        if ", IN" in temp or "Indiana".upper() in temp:
            states['IN'][1] += 1
        if ", KS" in temp or "Kansas".upper() in temp:
            states['KS'][1] += 1
        if ", KY" in temp or "Kentucky".upper() in temp:
            states['KY'][1] += 1
        if ", LA" in temp or "Louisiana".upper() in temp:
            states['LA'][1] += 1
        if ", ME" in temp or "Maine".upper() in temp:
            states['ME'][1] += 1
        if ", MD" in temp or "Maryland".upper() in temp:
            states['MD'][1] += 1
        if ", MA" in temp or "Massachusetts".upper() in temp:
            states['MA'][1] += 1
        if ", MI" in temp or "Michigan".upper() in temp:
            states['MI'][1] += 1
        if ", MN" in temp or "Minnesota".upper() in temp:
            states['MN'][1] += 1
        if ", MO" in temp or "Missouri".upper() in temp:
            states['MO'][1] += 1
        if ", MS" in temp or "Mississippi".upper() in temp:
            states['MS'][1] += 1
        if ", MT" in temp or "Montana".upper() in temp:
            states['MT'][1] += 1
        if ", NC" in temp or "North Carolina".upper() in temp:
            states['NC'][1] += 1
        if ", ND" in temp or "North Dakota".upper() in temp:
            states['ND'][1] += 1
        if ", NE" in temp or "Nebraska".upper() in temp:
            states['NE'][1] += 1
        if ", NH" in temp or "New Hampshire".upper() in temp:
            states['NH'][1] += 1
        if ", NJ" in temp or "New Jersey".upper() in temp:
            states['NJ'][1] += 1
        if ", NM" in temp or "New Mexico".upper() in temp:
            states['NM'][1] += 1
        if ", NV" in temp or "Nevada".upper() in temp:
            states['NV'][1]+= 1
        if ", NY" in temp or "New York".upper() in temp:
            states['NY'][1] += 1
        if ", OH" in temp or "Ohio".upper() in temp:
            states['OH'][1] += 1
        if ", OK" in temp or "Oklahoma".upper() in temp:
            states['OK'][1] += 1
        if ", OR" in temp or "Oregon".upper() in temp:
            states['OR'][1] += 1
        if ", PA" in temp or "Pennsylvania".upper() in temp:
            states['PA'][1] += 1
        if ", RI" in temp or "Rhode Island".upper() in temp:
            states['RI'][1] += 1
        if ", SC" in temp or "South Carolina".upper() in temp:
            states['SC'][1] += 1
        if ", SD" in temp or "South Dakota".upper() in temp:
            states['SD'][1] += 1
        if ", TN" in temp or "Tennessee".upper() in temp:
            states['TN'][1] += 1
        if ", TX" in temp or "Texas".upper() in temp:
            states['TX'][1] += 1
        if ", UT" in temp or "Utah".upper() in temp:
            states['UT'][1] += 1
        if ", VA" in temp or "Virginia".upper() in temp:
            states['VA'][1] += 1
        if ", VT" in temp or "Vermont".upper() in temp:
            states['VT'][1] += 1
        if ", WA" in temp or "Washington".upper() in temp:
            states['WA'][1] += 1
        if ", WI" in temp or "Wisconsin".upper() in temp:
            states['WI'][1] += 1
        if ", WV" in temp or "West Virginia".upper() in temp:
            states['WV'][1] += 1
        if ", WY" in temp or "Wyoming".upper() in temp:
            states['WY'][1] += 1
    #print s, tates
    #print (32, time.time())
    tup2 = []
    sum = 0
    for key, val in states.items():
        sum = sum + val[1]
    for key, val in states.items():
        tup2.append((key,val[0],val[1]*1.0/sum))
    #print(tup2)
        #csvWriter.writerow([key,val[0],val[1]])
    #print (42, time.time())
    dfm2 = pd.DataFrame(tup2, columns = ['code','state','activity'])
    #print(dfm2)
    #df = pd.read_csv('results.csv')

    for col in dfm2.columns:
        dfm2[col] = dfm2[col].astype(str)

    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
                [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
    dt['text'] = dfm2['state'] + '<br>' +\
        'TRENDS:'+'<br>'+df3['A']+'<br>'+\
        df3['B']+'<br>'+df3['C']+'<br>'+df3['D']+\
        '<br>'+df3['E']+'<br>'+df3['F']+'<br>'\
        +df3['G']+'<br>'+df3['H']+'<br>'+df3['I']+'<br>'+df3['J']

    data = [ dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = dfm2['code'],
            z = dfm2['activity'].astype(float),
            locationmode = 'USA-states',
            text = dt['text'],
            marker = dict(
                line = dict (
                    color = 'rgb(255,255,255)',
                    width = 1
                ) ),
            colorbar = dict(
                title = "Tweet Activities")
            ) ]


    return {
        'data': data,
        'layout': dict(
        title = 'Twitter Geographic Activities Ratio For Keyword "{}"'.format(state),
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
        }

@app.callback(
    Output('selected-data', 'figure'),
    [],[State('input', 'value')], [Event('submit', 'click')])
def update_graph1(state):

    states = dict()
    #print (12, time.time())
    with open('states.csv','r') as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        words = line.split(",")
        states[words[0]] = [words[1],0,int(words[2])]
# import sqlite3
    #print (22, time.time())
    conn = sqlite3.connect('twitter_old.db')
    #c = conn.cursor()
    temp = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ", conn ,params=('%' + state + '%',))
    #print my_id
    for index,row in temp.iterrows():
        temp = row['location'].upper()
        if ", AL" in temp or "Alabama".upper() in temp:
            states['AL'][1] += 1
        if ", AK" in temp or "Alaska".upper() in temp:
            states['AK'][1] += 1
        if ", AR" in temp or "Arkansas".upper() in temp:
            states['AR'][1] += 1
        if ", AZ" in temp or "Arizona".upper() in temp:
            states['AZ'][1] += 1
        if ", CA" in temp or "California".upper() in temp:
            states['CA'][1] += 1
        if ", CO" in temp or "Colorado".upper() in temp:
            states['CO'][1] += 1
        if ", CT" in temp or "Connecticut".upper() in temp:
            states['CT'][1] += 1
        if ", DE" in temp or "Delaware".upper() in temp:
            states['DE'][1] += 1
        if ", FL" in temp or "Florida".upper() in temp:
            states['FL'][1] += 1
        if ", GA" in temp or "Georgia".upper() in temp:
            states['GA'][1] += 1
        if ", HI" in temp or "Hawaii".upper() in temp:
            states['HI'][1] += 1
        if ", IA" in temp or "Iowa".upper() in temp:
            states['IA'][1] += 1
        if ", ID" in temp or "Idaho".upper() in temp:
            states['ID'][1] += 1
        if ", IL" in temp or "Illinois".upper() in temp:
            states['IL'][1] += 1
        if ", IN" in temp or "Indiana".upper() in temp:
            states['IN'][1] += 1
        if ", KS" in temp or "Kansas".upper() in temp:
            states['KS'][1] += 1
        if ", KY" in temp or "Kentucky".upper() in temp:
            states['KY'][1] += 1
        if ", LA" in temp or "Louisiana".upper() in temp:
            states['LA'][1] += 1
        if ", ME" in temp or "Maine".upper() in temp:
            states['ME'][1] += 1
        if ", MD" in temp or "Maryland".upper() in temp:
            states['MD'][1] += 1
        if ", MA" in temp or "Massachusetts".upper() in temp:
            states['MA'][1] += 1
        if ", MI" in temp or "Michigan".upper() in temp:
            states['MI'][1] += 1
        if ", MN" in temp or "Minnesota".upper() in temp:
            states['MN'][1] += 1
        if ", MO" in temp or "Missouri".upper() in temp:
            states['MO'][1] += 1
        if ", MS" in temp or "Mississippi".upper() in temp:
            states['MS'][1] += 1
        if ", MT" in temp or "Montana".upper() in temp:
            states['MT'][1] += 1
        if ", NC" in temp or "North Carolina".upper() in temp:
            states['NC'][1] += 1
        if ", ND" in temp or "North Dakota".upper() in temp:
            states['ND'][1] += 1
        if ", NE" in temp or "Nebraska".upper() in temp:
            states['NE'][1] += 1
        if ", NH" in temp or "New Hampshire".upper() in temp:
            states['NH'][1] += 1
        if ", NJ" in temp or "New Jersey".upper() in temp:
            states['NJ'][1] += 1
        if ", NM" in temp or "New Mexico".upper() in temp:
            states['NM'][1] += 1
        if ", NV" in temp or "Nevada".upper() in temp:
            states['NV'][1]+= 1
        if ", NY" in temp or "New York".upper() in temp:
            states['NY'][1] += 1
        if ", OH" in temp or "Ohio".upper() in temp:
            states['OH'][1] += 1
        if ", OK" in temp or "Oklahoma".upper() in temp:
            states['OK'][1] += 1
        if ", OR" in temp or "Oregon".upper() in temp:
            states['OR'][1] += 1
        if ", PA" in temp or "Pennsylvania".upper() in temp:
            states['PA'][1] += 1
        if ", RI" in temp or "Rhode Island".upper() in temp:
            states['RI'][1] += 1
        if ", SC" in temp or "South Carolina".upper() in temp:
            states['SC'][1] += 1
        if ", SD" in temp or "South Dakota".upper() in temp:
            states['SD'][1] += 1
        if ", TN" in temp or "Tennessee".upper() in temp:
            states['TN'][1] += 1
        if ", TX" in temp or "Texas".upper() in temp:
            states['TX'][1] += 1
        if ", UT" in temp or "Utah".upper() in temp:
            states['UT'][1] += 1
        if ", VA" in temp or "Virginia".upper() in temp:
            states['VA'][1] += 1
        if ", VT" in temp or "Vermont".upper() in temp:
            states['VT'][1] += 1
        if ", WA" in temp or "Washington".upper() in temp:
            states['WA'][1] += 1
        if ", WI" in temp or "Wisconsin".upper() in temp:
            states['WI'][1] += 1
        if ", WV" in temp or "West Virginia".upper() in temp:
            states['WV'][1] += 1
        if ", WY" in temp or "Wyoming".upper() in temp:
            states['WY'][1] += 1
    #print s, tates
    #print (32, time.time())
    tup1 = []
    # for key, val in states.items():
    #     sum = sum + val[1]
    for key, val in states.items():
        tup1.append((key,val[0],val[1]*1.0/val[2]))
    print(tup1)
        #csvWriter.writerow([key,val[0],val[1]])
    #print (42, time.time())
    dfm1 = pd.DataFrame(tup1, columns = ['code','state','activity'])
    #print(df.activity.sort_values())
    #df['activity'] = df['activity'].astype(int)
    dfm1 = dfm1.sort_values('activity')


    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
                [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
    dt['text'] = dfm1['state'] + '<br>' +\
        'TRENDS:'+'<br>'+df3['A']+'<br>'+\
        df3['B']+'<br>'+df3['C']+'<br>'+df3['D']+\
        '<br>'+df3['E']+'<br>'+df3['F']+'<br>'\
        +df3['G']+'<br>'+df3['H']+'<br>'+df3['I']+'<br>'+df3['J']

    data = [ dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = dfm1['code'],
            z = dfm1['activity'].astype(float),
            locationmode = 'USA-states',
            text = dt['text'],
            marker = dict(
                line = dict (
                    color = 'rgb(255,255,255)',
                    width = 1
                ) ),
            colorbar = dict(
                title = "Tweet Activities")
            ) ]


    fig = dict(data=data, layout=dict(
        title = 'Twitter Geographic Activities for keyword "{}"'.format(state),
             ))
    fig1 = dict(data = [go.Bar(x=dfm1['state'], y=dfm1['activity'])],
				layout = dict(
                    title='Twitter Activities/ Population From each State For Keyword "{}"<br> (Tweets/ Per 1,000 people)'.format(state),
					paper_bgcolor = '#F4F4F8',
					plot_bgcolor = '#F4F4F8',
					height = 600
				)
			)
    return fig1

if __name__ == '__main__':
    app.run_server(port = 8080, host = '127.0.0.1')
