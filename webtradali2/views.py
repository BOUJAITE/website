from django.shortcuts import render
from django.http import HttpResponse
from .forms import quotation, mlmodel
#PACKAGES
import requests
import threading, sys
import bs4
import json
import pprint
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from datetime import datetime
import pandas_datareader.data as web
import numpy as np
import chart_studio.plotly as py
from scipy import stats

def index(request):

	#date
	start_date = '2010-1-1'
	now = datetime.now()
	present_year = now.year
	present_month = now.month
	present_day = now.day
	end_date = '%d-%d-%d'%(present_year,present_month,present_day)
	
	#DATAFRAME
	df = web.DataReader('^FCHI' , 'yahoo', start_date, end_date)
	monthlydf = df.resample('M').apply(lambda x: x[-1])
	quartersdf = df.resample('4M').apply(lambda x: x[-1])
	
	#market
	df_market = web.DataReader('^FCHI' , 'yahoo', start_date, end_date)
	monthly_df_market = df_market.resample('M').apply(lambda x: x[-1])
	quarters_df_market = df_market.resample('4M').apply(lambda x: x[-1])


	################################ FUNCTIONS 


	def rolling_mean_std(df,df_close,t_window) :
		df[f'{t_window}'] = df_close.rolling(window=t_window).mean()
		rm = df_close.rolling(window=t_window).mean()
		df[f'std_{t_window}'] = df_close.rolling(window=t_window).std()
		rstd = df_close.rolling(window=t_window).std()
		df['upper_band'] = rm + rstd *2
		df['lower_band'] = rm - rstd *2
		return df

	def get_estimator(price_data, window=30, trading_periods=252, clean=True):
		'''
		Volatility calculation
		'''

		log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
		log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

		rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
		
		def f(v):
			return (trading_periods * v.mean())**0.5
		
		result = rs.rolling(window=window, center=False).apply(func=f)
		
		if clean:
			return result.dropna()
		else:
			return result

		
	def moving_average(df, n):
		n=n+1
		MA = pd.Series(df['Adj Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n-1))
		df= df.join(MA)
		return df


	def exponential_moving_average(df, n):
		n=n+1
		EMA = pd.Series(df['Adj Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n-1))
		df= df.join(EMA)
		return df


	def compute_daily_returns(df):
		daily_returns = (df/df.shift(1)) - 1
		daily_returns = daily_returns.fillna(0)
		return daily_returns


	def relative_strength_index(df, n):
		'''
		RSI à utiliser dans le DL
		'''
		n=n+1
		i = 0
		UpI = [0]
		DoI = [0]
		while i + 1 <= df.reset_index(drop = True).index[-1]:
			UpMove = df.reset_index(drop = True).loc[i + 1, 'High'] - df.reset_index(drop = True).loc[i, 'High']
			DoMove = df.reset_index(drop = True).loc[i, 'Low'] - df.reset_index(drop = True).loc[i + 1, 'Low']
			if UpMove > DoMove and UpMove > 0:
				UpD = UpMove
			else:
				UpD = 0
			UpI.append(UpD)
			if DoMove > UpMove and DoMove > 0:
				DoD = DoMove
			else:
				DoD = 0
			DoI.append(DoD)
			i = i + 1
		UpI = pd.Series(UpI)
		DoI = pd.Series(DoI)
		PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
		NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
		RSI = pd.Series(PosDI / (PosDI + NegDI))
		return RSI

	def VaR(df):
		daily_returns = df["Adj Close"].pct_change()
		daily_returns = daily_returns.dropna(how='any') 
		
		#Sort Returns in Ascending Order
		sorted_returns = sorted(daily_returns)
		varg = np.percentile(sorted_returns, 5)
		return varg

	#indicators
	
	df['daily returns'] = compute_daily_returns(df['Adj Close'])
	#########indicators
	for n in [5,10,20,50,100]:
		##########moving averages
		df = moving_average(df,n)
		df = exponential_moving_average(df,n)
		##########RSI
		df[f'RSI{n}'] = relative_strength_index(df,n).values
	
	########CAPM (MEDAF)

	##########monthly_data
	monthly_returns = monthlydf['Adj Close'].pct_change(1)
	clean_monthly_returns = monthly_returns.dropna(axis=0)# drop first missing row
	y = clean_monthly_returns

	market_monthly_returns = monthly_df_market['Adj Close'].pct_change(1)
	market_clean_monthly_returns = market_monthly_returns.dropna(axis=0)
	x = market_clean_monthly_returns
	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

	df['beta'] = slope


	######## Valuation at risk VaR
	df['VaR'] = VaR(df)

	########sharp ratio
	returns = df['daily returns']
	# annualized Sharpe ratio
	sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
	df['sharpe_ratio'] = sharpe_ratio

	######## compound annual growth rate (CAGR)
	# Get the number of days 
	days = (df.index[-1] - df.index[0]).days
	# Calculate the CAGR 
	cagr = ((((df['Adj Close'][-1]) / df['Adj Close'][1])) ** (365.0/days)) - 1
	df['cagr'] = cagr

	#param
	ma5 = "{0:.2f}".format(round(df['MA_5'].iloc[-1],2)) 
	ma10 = "{0:.2f}".format(round(df['MA_10'].iloc[-1],2))  
	ma20 = "{0:.2f}".format(round(df['MA_20'].iloc[-1],2))   
	ma50 = "{0:.2f}".format(round(df['MA_50'].iloc[-1],2))   
	ma100 = "{0:.2f}".format(round(df['MA_100'].iloc[-1],2))   

	ema5 = "{0:.2f}".format(round(df['EMA_5'].iloc[-1],2)) 
	ema10 = "{0:.2f}".format(round(df['EMA_10'].iloc[-1],2))  
	ema20 = "{0:.2f}".format(round(df['EMA_20'].iloc[-1],2))   
	ema50 = "{0:.2f}".format(round(df['EMA_50'].iloc[-1],2))   
	ema100 = "{0:.2f}".format(round(df['EMA_100'].iloc[-1],2))

	rsi5 = "{0:.3f}".format(round(df['RSI5'].iloc[-1],2))
	rsi10 = "{0:.3f}".format(round(df['RSI10'].iloc[-1],2))		
	rsi20 = "{0:.3f}".format(round(df['RSI20'].iloc[-1],2))
	rsi50 = "{0:.3f}".format(round(df['RSI50'].iloc[-1],2))		
	rsi100 = "{0:.3f}".format(round(df['RSI100'].iloc[-1],2))

	beta = 	"{0:.3f}".format(round(df['beta'].iloc[-1],2))
	var = "{0:.3f}".format(round(df['VaR'].iloc[-1],2))
	sr = "{0:.3f}".format(round(df['sharpe_ratio'].iloc[-1],2))
	cagr = "{0:.3f}".format(round(df['cagr'].iloc[-1],2))

	def action_quote(ma1,ma2):
	    if ma1 > ma2: action = "BUY"
	    elif ma1 < ma2: action = "SELL"
	    else: action = "---"
	    return action

	def action_rsi(value):
	    value = float(value)
	    if 0 <= value <= 0.399: act="BUY"
	    elif 0.7 <= value <= 1: act = "SELL"
	    else: act = "---"
	    return act

	ma5_10 = action_quote(ma5, ma10)
	ma20_50 = action_quote(ma20, ma50)
	ema5_10 = action_quote(ema5, ema10)
	ema20_50 = action_quote(ema20, ema50)


	act5 = action_rsi(rsi5)
	act10 = action_rsi(rsi10)
	act20 = action_rsi(rsi20)
	act50 = action_rsi(rsi50)

	def bbands(price, window_size=10, num_of_std=5):
		rolling_mean = price.rolling(window=window_size).mean()
		rolling_std  = price.rolling(window=window_size).std()
		upper_band = rolling_mean + (rolling_std*num_of_std)
		lower_band = rolling_mean - (rolling_std*num_of_std)
		return rolling_mean, upper_band, lower_band

	def movingaverage(interval, window_size=10):
		window = np.ones(int(window_size))/float(window_size)
		return np.convolve(interval, window, 'same')

	mv_y = movingaverage(df.Close)
	mv_x = df.index

	# Clip the ends
	mv_x = mv_x[5:-5]
	mv_y = mv_y[5:-5]

	bb_avg, bb_upper, bb_lower = bbands(df.Close)
	#GRAPHIC

	x,y,z,w,k,v=[],[],[],[],[],[]

	x=df.index.tolist()
	y=df['Open'].tolist()
	z=df['High'].tolist()
	w=df['Low'].tolist()
	k=df['Close'].tolist()
	v=df['Volume'].tolist()
	bb_avg = bb_avg.tolist()
	bb_upper = bb_upper.tolist()
	bb_lower = bb_lower.tolist()
	mv_x = mv_x.tolist()
	mv_y = mv_y.tolist()


	trace1 = go.Candlestick(
		x=x,
		open = y,
		high = z,
		low = w,
		close = k,
		legendgroup='Candlestick',
		name='Candlestick'
		#whiskerwidth = 1
		)

	INCREASING_COLOR = '#78d89c'
	DECREASING_COLOR = '#ff7373'
	colors = []

	for i in range(len(df.Close)):
		if i != 0:
			if df.Close[i] > df.Close[i-1]:
				colors.append(INCREASING_COLOR)
			else:
				colors.append(DECREASING_COLOR)
		else:
			colors.append(DECREASING_COLOR)

	trace3 = go.Scatter(
		x=mv_x,
		y=mv_y,
		mode='lines',
		line=dict(width=1),
		marker_color='#E377C2',
		marker_size=1,
		name='moving average'
		)

	trace4 = go.Scatter(
		x=x,
		y=bb_upper,
		mode='lines',
		marker_color='#ccc', 
		line=dict(width=1),
		marker_size=1,
		name='upper band'
		)

	trace5 = go.Scatter(
		x=x,
		y=bb_lower,
		mode='lines',
		marker_color='#ccc', 
		line=dict(width=1),
		name='lower band'
		)

	trace2 = go.Bar(
		x=x,
		y=v,
		marker=dict(line=dict( color=colors , width=1 )),
		legendgroup='Volume',
		name='Volume',
		opacity=0.5,
		yaxis='y2'
		)
	layout = go.Layout(
		# autosize=True,
		# width = 800,
		# height=900,
		xaxis=dict(
			rangeslider=dict(visible=True)
		),
		yaxis=dict(
			title="Price",
			showticklabels = True,
			titlefont=dict(
				color="#DDDDDD"
			),
			tickfont=dict(
				color="#DDDDDD"
			),
		),
		yaxis2=dict(
			title="Volume",
			titlefont=dict(
				color="#DDDDDD"
			),
			tickfont=dict(
				color="#DDDDDD"
			),
			anchor="free",
			side="right",
			position=1,
			overlaying= "y"
		),
		legend=dict(
			orientation = 'h',
			y=1,
			x=0,
			yanchor='bottom'
		)
	)

	data = go.Data([trace1,trace2,trace3,trace4,trace5])
	figure = go.Figure(data, layout)



	plot_div = plot(figure, output_type='div', include_plotlyjs=False)

				
	#SCRAP PARAMETERS
	TEMPS_ATTENTE = 10
	IDENT_USER_AGENT_2018 = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0'
	en_tetes = {'User-Agent': IDENT_USER_AGENT_2018,}
	id_action = ['386','389','412','6910','7019','393','396','397','390','404'
	             ,'407','394','6957','395','405','6987','670','400','7059'
	             ,'399','411','409','406','402','6979','419','6991','403','414'
	             ,'401','410','6981','417','7006','6952','416','420','418'
	             ,'415','671']
	sess = requests.Session()


	#SCRAP

	patriot = True

	def tradingscrap():
	    if patriot:
	        #threading.Timer(TEMPS_ATTENTE, tradingscrap).start()
	        res = sess.get('https://www.investing.com/indices/france-40-components', headers=en_tetes)
	        soup = bs4.BeautifulSoup(res.content, 'html.parser')
	        html_actions = soup.find_all('td',attrs={'class':'plusIconTd'})
	        label_actions = re.findall(r'''data-name=\"(\w*\s*\W*\w*\s*\w*\s*\w*\s*\w*\s*\w*)\"''',str(html_actions))
	        
	        reponses=[]
	        performance = soup.find_all('table',attrs={'id':'cr1'})
	        perf = re.findall(r"\>(-?\+?\d+\.\d*%)\<",str(performance))

	        for i in id_action: 
	            reponses += soup.find_all('td',attrs={'class':'pid-%s-last'%(i)})


	        cotations = re.findall(r"\>(\d*\.\d*)\<",str(reponses))
	        d = {'label': label_actions,'cotation': cotations,'performance':perf}
	        df = pd.DataFrame(data=d)
	        df['cotation']= pd.to_numeric(df['cotation'])
	        df['cotation'] = df['cotation'].map('{:,.2f}'.format)
	        print(df)
	        return df
	    else:
	        print('arret du scrap')
	        sys.exit

	df= tradingscrap()

	switch = {'LVMH Moet Hennessy Louis Vuitton SE':'LVMH SE',"Dassault Systemes SE":"Dassault Systemes",'Compagnie Generale des Etablissements Michelin SCA':'Michelin SCA','WFD Unibail Rodamco NV':'Unibail NV','Veolia Environnement VE SA':'Veolia SA','Compagnie de Saint Gobain SA':'Saint Gobain SA','STMicroelectronics NV':'STMicroelectronics','Hermes International SCA':'Hermes SCA'}
	label = df['label']
	for key, value in switch.items():
		label = label.replace(key,value)
	cotation = df['cotation']
	perf = df['performance']
	class_cell = []
	for word in perf:
		if re.match(r'-',word):
			class_cell.append("Negatif")
		else:
			class_cell.append("Positif")
	dict_cot = zip(label, cotation, perf, class_cell)
	
	
	return render(request, 'webtradali2/index.html', locals())

def about(request):
	
	return render(request, 'webtradali2/about.html', locals())

def tech_analysis(request):

	#date
	start_date = '2010-1-1'
	now = datetime.now()
	present_year = now.year
	present_month = now.month
	present_day = now.day
	end_date = '%d-%d-%d'%(present_year,present_month,present_day)
	
	#validation
	selec = quotation(request.POST or None, request.FILES or None)
	if selec.is_valid():
		cote = selec.cleaned_data["name_quote"]
		#DATAFRAME
		df = web.DataReader(f'{cote}' , 'yahoo', start_date, end_date)
		monthlydf = df.resample('M').apply(lambda x: x[-1])
		quartersdf = df.resample('4M').apply(lambda x: x[-1])
		
		#market
		df_market = web.DataReader('^FCHI' , 'yahoo', start_date, end_date)
		monthly_df_market = df_market.resample('M').apply(lambda x: x[-1])
		quarters_df_market = df_market.resample('4M').apply(lambda x: x[-1])


		################################ FUNCTIONS 


		def rolling_mean_std(df,df_close,t_window) :
			df[f'{t_window}'] = df_close.rolling(window=t_window).mean()
			rm = df_close.rolling(window=t_window).mean()
			df[f'std_{t_window}'] = df_close.rolling(window=t_window).std()
			rstd = df_close.rolling(window=t_window).std()
			df['upper_band'] = rm + rstd *2
			df['lower_band'] = rm - rstd *2
			return df

		def get_estimator(price_data, window=30, trading_periods=252, clean=True):
			'''
			Volatility calculation
			'''

			log_hl = (price_data['High'] / price_data['Low']).apply(np.log)
			log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

			rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
			
			def f(v):
				return (trading_periods * v.mean())**0.5
			
			result = rs.rolling(window=window, center=False).apply(func=f)
			
			if clean:
				return result.dropna()
			else:
				return result

			
		def moving_average(df, n):
			n=n+1
			MA = pd.Series(df['Adj Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n-1))
			df= df.join(MA)
			return df


		def exponential_moving_average(df, n):
			n=n+1
			EMA = pd.Series(df['Adj Close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n-1))
			df= df.join(EMA)
			return df


		def compute_daily_returns(df):
			daily_returns = (df/df.shift(1)) - 1
			daily_returns = daily_returns.fillna(0)
			return daily_returns


		def relative_strength_index(df, n):
			'''
			RSI à utiliser dans le DL
			'''
			n=n+1
			i = 0
			UpI = [0]
			DoI = [0]
			while i + 1 <= df.reset_index(drop = True).index[-1]:
				UpMove = df.reset_index(drop = True).loc[i + 1, 'High'] - df.reset_index(drop = True).loc[i, 'High']
				DoMove = df.reset_index(drop = True).loc[i, 'Low'] - df.reset_index(drop = True).loc[i + 1, 'Low']
				if UpMove > DoMove and UpMove > 0:
					UpD = UpMove
				else:
					UpD = 0
				UpI.append(UpD)
				if DoMove > UpMove and DoMove > 0:
					DoD = DoMove
				else:
					DoD = 0
				DoI.append(DoD)
				i = i + 1
			UpI = pd.Series(UpI)
			DoI = pd.Series(DoI)
			PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
			NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
			RSI = pd.Series(PosDI / (PosDI + NegDI))
			return RSI

		def VaR(df):
			daily_returns = df["Adj Close"].pct_change()
			daily_returns = daily_returns.dropna(how='any') 
			
			#Sort Returns in Ascending Order
			sorted_returns = sorted(daily_returns)
			varg = np.percentile(sorted_returns, 5)
			return varg

		#indicators
		
		df['daily returns'] = compute_daily_returns(df['Adj Close'])
		#########indicators
		for n in [5,10,20,50,100]:
			##########moving averages
			df = moving_average(df,n)
			df = exponential_moving_average(df,n)
			##########RSI
			df[f'RSI{n}'] = relative_strength_index(df,n).values
		
		########CAPM (MEDAF)

		##########monthly_data
		monthly_returns = monthlydf['Adj Close'].pct_change(1)
		clean_monthly_returns = monthly_returns.dropna(axis=0)# drop first missing row
		y = clean_monthly_returns

		market_monthly_returns = monthly_df_market['Adj Close'].pct_change(1)
		market_clean_monthly_returns = market_monthly_returns.dropna(axis=0)
		x = market_clean_monthly_returns
		slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

		df['beta'] = slope


		######## Valuation at risk VaR
		df['VaR'] = VaR(df)

		########sharp ratio
		returns = df['daily returns']
		# annualized Sharpe ratio
		sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
		df['sharpe_ratio'] = sharpe_ratio

		######## compound annual growth rate (CAGR)
		# Get the number of days 
		days = (df.index[-1] - df.index[0]).days
		# Calculate the CAGR 
		cagr = ((((df['Adj Close'][-1]) / df['Adj Close'][1])) ** (365.0/days)) - 1
		df['cagr'] = cagr
		
		#param
		ma5 = "{0:.2f}".format(round(df['MA_5'].iloc[-1],2)) 
		ma10 = "{0:.2f}".format(round(df['MA_10'].iloc[-1],2))  
		ma20 = "{0:.2f}".format(round(df['MA_20'].iloc[-1],2))   
		ma50 = "{0:.2f}".format(round(df['MA_50'].iloc[-1],2))   
		ma100 = "{0:.2f}".format(round(df['MA_100'].iloc[-1],2))   

		ema5 = "{0:.2f}".format(round(df['EMA_5'].iloc[-1],2)) 
		ema10 = "{0:.2f}".format(round(df['EMA_10'].iloc[-1],2))  
		ema20 = "{0:.2f}".format(round(df['EMA_20'].iloc[-1],2))   
		ema50 = "{0:.2f}".format(round(df['EMA_50'].iloc[-1],2))   
		ema100 = "{0:.2f}".format(round(df['EMA_100'].iloc[-1],2))

		rsi5 = "{0:.3f}".format(round(df['RSI5'].iloc[-1],2))
		rsi10 = "{0:.3f}".format(round(df['RSI10'].iloc[-1],2))		
		rsi20 = "{0:.3f}".format(round(df['RSI20'].iloc[-1],2))
		rsi50 = "{0:.3f}".format(round(df['RSI50'].iloc[-1],2))		
		rsi100 = "{0:.3f}".format(round(df['RSI100'].iloc[-1],2))

		beta = 	"{0:.3f}".format(round(df['beta'].iloc[-1],2))
		var = "{0:.3f}".format(round(df['VaR'].iloc[-1],2))
		sr = "{0:.3f}".format(round(df['sharpe_ratio'].iloc[-1],2))
		cagr = "{0:.3f}".format(round(df['cagr'].iloc[-1],2))

		def action_quote(ma1,ma2):
		    if ma1 > ma2: action = "BUY"
		    elif ma1 < ma2: action = "SELL"
		    else: action = "---"
		    return action

		def action_rsi(value):
		    value = float(value)
		    if 0 <= value <= 0.3: act="BUY"
		    elif 0.7 <= value <= 1: act = "SELL"
		    else: act = "---"
		    return act

		ma5_10 = action_quote(ma5, ma10)
		ma20_50 = action_quote(ma20, ma50)
		ema5_10 = action_quote(ema5, ema10)
		ema20_50 = action_quote(ema20, ema50)


		act5 = action_rsi(rsi5)
		act10 = action_rsi(rsi10)
		act20 = action_rsi(rsi20)
		act50 = action_rsi(rsi50)

		def bbands(price, window_size=10, num_of_std=5):
			rolling_mean = price.rolling(window=window_size).mean()
			rolling_std  = price.rolling(window=window_size).std()
			upper_band = rolling_mean + (rolling_std*num_of_std)
			lower_band = rolling_mean - (rolling_std*num_of_std)
			return rolling_mean, upper_band, lower_band

		def movingaverage(interval, window_size=10):
			window = np.ones(int(window_size))/float(window_size)
			return np.convolve(interval, window, 'same')
		mv_y = movingaverage(df.Close)
		mv_x = df.index

		# Clip the ends
		mv_x = mv_x[5:-5]
		mv_y = mv_y[5:-5]

		bb_avg, bb_upper, bb_lower = bbands(df.Close)
		#GRAPHIC

		x,y,z,w,k,v=[],[],[],[],[],[]

		x=df.index.tolist()
		y=df['Open'].tolist()
		z=df['High'].tolist()
		w=df['Low'].tolist()
		k=df['Close'].tolist()
		v=df['Volume'].tolist()
		bb_avg = bb_avg.tolist()
		bb_upper = bb_upper.tolist()
		bb_lower = bb_lower.tolist()
		mv_x = mv_x.tolist()
		mv_y = mv_y.tolist()


		trace1 = go.Candlestick(
			x=x,
			open = y,
			high = z,
			low = w,
			close = k,
			legendgroup='Candlestick',
			name='Candlestick'
			#whiskerwidth = 1
			)

		INCREASING_COLOR = '#78d89c'
		DECREASING_COLOR = '#ff7373'
		colors = []

		for i in range(len(df.Close)):
			if i != 0:
				if df.Close[i] > df.Close[i-1]:
					colors.append(INCREASING_COLOR)
				else:
					colors.append(DECREASING_COLOR)
			else:
				colors.append(DECREASING_COLOR)

		trace3 = go.Scatter(
			x=mv_x,
			y=mv_y,
			mode='lines',
			line=dict(width=1),
			marker_color='#E377C2',
			marker_size=1,
			name='moving average'
			)

		trace4 = go.Scatter(
			x=x,
			y=bb_upper,
			mode='lines',
			marker_color='#ccc', 
			line=dict(width=1),
			marker_size=1,
			name='upper band'
			)

		trace5 = go.Scatter(
			x=x,
			y=bb_lower,
			mode='lines',
			marker_color='#ccc', 
			line=dict(width=1),
			name='lower band'
			)

		trace2 = go.Bar(
			x=x,
			y=v,
			marker=dict(line=dict( color=colors , width=1 )),
			legendgroup='Volume',
			name='Volume',
			opacity=0.5,
			yaxis='y2'
			)
		layout = go.Layout(
			# autosize=True,
			# width = 800,
			# height=900,
			xaxis=dict(
				rangeslider=dict(visible=True)
			),
			yaxis=dict(
				title="Price",
				showticklabels = True,
				titlefont=dict(
					color="#DDDDDD"
				),
				tickfont=dict(
					color="#DDDDDD"
				),
			),
			yaxis2=dict(
				title="Volume",
				titlefont=dict(
					color="#DDDDDD"
				),
				tickfont=dict(
					color="#DDDDDD"
				),
				anchor="free",
				side="right",
				position=1,
				overlaying= "y"
			),
			legend=dict(
				orientation = 'h',
				y=1,
				x=0,
				yanchor='bottom'
			)
		)

		data = go.Data([trace1,trace2,trace3,trace4,trace5])
		figure = go.Figure(data, layout)



		plot_div = plot(figure, output_type='div', include_plotlyjs=False)
	return render(request, 'webtradali2/ta.html', locals())









def price_pred(request):
	pred = mlmodel(request.POST or None, request.FILES or None)
	if pred.is_valid():
		mode = pred.cleaned_data["name_models"]
		quotation_name = pred.cleaned_data["name_quote"]
		#%%Packages
		import pandas as pd
		import numpy as np
		from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
		from sklearn.metrics import accuracy_score
		from sklearn.model_selection import ParameterGrid
		import matplotlib.pyplot as plt
		from sklearn.metrics import classification_report, roc_curve, auc, mean_absolute_error, mean_squared_error
		#%%Data
		df = pd.read_excel(fr'C:\Users\Boujaite Ali\Desktop\projet fin etude\DWH\{quotation_name}.xlsx')
		df_prep = df[['Open','Close','High','Low','Volume','Adj Close']]
		# Features construction 
		df_prep['Open-Close'] = (df_prep.Open - df_prep.Close)/df_prep.Open
		df_prep['High-Low'] = (df_prep.High - df_prep.Low)/df_prep.Low
		df_prep['percent_change'] = df_prep['Adj Close'].pct_change()
		df_prep['std_5'] = df_prep['percent_change'].rolling(5).std()
		df_prep['ret_5'] = df_prep['percent_change'].rolling(5).mean()
		df_prep.dropna(inplace=True)
		feature_names = ['Open-Close', 'High-Low', 'std_5', 'ret_5']
		# X is the input variable
		X = df_prep[['Open-Close', 'High-Low', 'std_5', 'ret_5']]
		# Y is the target or output variable
		y = np.where(df_prep['Adj Close'].shift(-1) > df_prep['Adj Close'], 1, -1)
		# Total dataset length
		dataset_length = df_prep.shape[0]

		# Training dataset length
		split = int(dataset_length * 0.75)
		# Splitiing the X and y into train and test datasets
		X_train, X_test = X[:split], X[split:]
		y_train, y_test = y[:split], y[split:]

		if mode == "rf":
			rf = RandomForestClassifier(n_estimators=4,
											criterion='gini',
											max_depth=11,
											max_features=4,
											min_samples_split=10,
											min_samples_leaf=5,
											random_state=105)

			model_rf = rf.fit(X_train, y_train)
			score1 = "Accuracy score (training ): {0:.3f} %".format(rf.score(X_train, y_train)*100)
			s = accuracy_score(y_test, model_rf.predict(X_test), normalize=True)*100.0
			score2 = "Accuracy score (validation ): {0:.3f} %".format(s)
			last_pred = model_rf.predict(X_test)[-1]
			if last_pred == 1:
				conclusion = "The machine learning model (Random Forest) we suggest to BUY"
			if last_pred == -1:
				conclusion = "The machine learning model (Random Forest) we suggest to SELL"

		if mode == "gb":
			gb = GradientBoostingClassifier(n_estimators=3, learning_rate = 1, max_features=1, max_depth = 12, random_state = 108)
			model_gb = gb.fit(X_train, y_train)
			score1 = "Accuracy score (training ): {0:.3f} %".format(gb.score(X_train, y_train)*100)
			s = accuracy_score(y_test, model_gb.predict(X_test), normalize=True)*100.0
			score2 = "Accuracy score (validation ): {0:.3f} %".format(s)
			last_pred = model_gb.predict(X_test)[-1]
			if last_pred == 1:
				conclusion = "The machine learning model (Gradient Boosting) we suggest to BUY"
			if last_pred == -1:
				conclusion = "The machine learning model (Gradient Boosting) we suggest to SELL"


	return render(request, 'webtradali2/pp.html', locals())

def contact(request):
	
	return render(request, 'webtradali2/contact.html', locals())

def sum(request):
	
	return render(request, 'webtradali2/summury.html', locals())