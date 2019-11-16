# Twitter Map & Sentiments (Light Version)

[![demo](http://img.youtube.com/vi/9sqpyDrouSc/0.jpg)](http://www.youtube.com/watch?v=9sqpyDrouSc)

Clike above image to view the demo video (light version)

In this folder, `states_woeid.csv`, `states.csv`, `test.csv` and `trends_data.csv` are data source files to help organize and analyze our data. `combine.py` is the main python script that will run a web server of our app. `fetch_live.py` is the file that we use to fetch live steeam data. `fetch_trends.py` is the file that we use to fetch top-10 trends for each states. We didn't present this feature in presentation because of time, but we think it's an useful feature. To see the top trends, just hover the moust to a state. `twitter_new.db` is the database we use to store our live streaming data. In order to see live streaming feature, you much run `fetch_new.py` to stream data to the database. `fetch_map.py` is the file we use to fetch live data to our twitter map database. We don't recommend run this when deploy our app locally as its file size will grow up really fast. `twitrer_old.db` is the database we use for the twitter map. Note that the file on our AWS server is very large. This one is relatively small which contains less data.

## Getting Started (Packages Installation)

cd into the folder and run:
```
python3 -m venv twitter_venv
```
to create a new python3 virtual environment
```
source twitter_venv/bin/activate
```
to activate the environment
```
pip3 install sqlite3
pip3 install tweet-preprocessor
pip3 install vaderSentiment
pip3 install dash dash-renderer dash-html-components dash-core-components plotly
pip3 install pandas
pip3 install datetime
pip3 install numpy
pip3 install unidecode
pip3 install time
```
## Start Fetching Data

To fetch data for live sentiment analysis, run:
```
python3 fetch_new.py
```
And it will run unless you `ctrl+c` to stop it.

To fetch data for top-10 trends in each state, open another termimal window, get into the virtual environment as stated above, then run:
```
python3 fetch_trends.py
```
Note that this will automatically run every 5 minutes since twitter updates the geo trends data every 5 mintues.



## Run the Map App

To start local server and use the app, open another termimal window, get into the virtual environment as stated above, then run:
```
python3 combine.py
```
And a local web server should then be running at `127.0.0.1:8080`
Note that you may get `port already in use` error. In such case, you can open `combine.py` with any text editor of your choice, go to the last line of code and type in the port you want to use.
