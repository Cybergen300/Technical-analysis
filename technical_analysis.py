#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:40:30 2018

"""

import pandas as pd 
import matplotlib.pylab as plt
import numpy as np
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import datetime
import copy
from scipy.signal import savgol_filter as smooth


yf.pdr_override() 

stocks = ["BTCUSD=X"]
start = datetime.datetime(2010,8,1)
end = datetime.datetime.today().strftime('%Y-%m-%d')

data = pdr.get_data_yahoo(stocks, start=start, end=end)

#review the data 

plt.close('all')
fig, ax =plt.subplots(1)
ax.plot(data['Adj Close'], 'b', lw = 1.5, label = ' BTC/USD')
fig.autofmt_xdate()
plt.title('Historic of BTC price (2010 - 2018)')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('prices ($)')
plt.xlabel('dates')
plt.show()

#First we calculate the volatility 
#to do so we need to get the daily return of our asset 

close = data['Adj Close']
high = data['High']
low = data['Low']
Open = data['Open']


closeR = np.log(close[:-1] / close[1:].values) 


MEAN_RETURN = np.mean(closeR)

tab1 = {'close' : pd.Series(close),'high' : pd.Series(high), 'low' : pd.Series(low), 
        'closeR' : pd.Series(closeR), 'open' : pd.Series(Open)}


tab1_df = pd.DataFrame(tab1, columns= ['close', 'high','low','open', 'closeR'])
tab1_df['closeR'] = tab1_df['closeR'].shift(1)



#################
#academic_volat :
#################

def academic_vol(price_return, day_base): 
    volat = np.sqrt(np.sum((price_return-MEAN_RETURN)**2)/(len(price_return)-2))
    ann_acad_volat = np.sqrt(day_base) * volat
    return ann_acad_volat

academic_vol(tab1_df['closeR'], 365)


#####################
#trading volatility :
#####################
 
def trading_vol(price_return, day_base): 
    trade_volatility = np.sqrt(np.sum((price_return)**2)/(len(price_return)-2))
    ann_trade_volatility = np.sqrt(day_base) * trade_volatility
    return ann_trade_volatility 

trading_vol(tab1_df['closeR'],365)

#######################
#Parkinson volatility : 
#######################

def Parkinson_vol(high, low, day_base):
    PARKINSON = (1/ (4* len(high) *np.log(2)))
    volat_park = np.sqrt( PARKINSON * np.sum( np.log(high/low) **2 ))
    ann_volat_park = np.sqrt(day_base) * volat_park
    return ann_volat_park 

Parkinson_vol(tab1_df['high'], tab1_df['low'], 365)


#########################
#Garman Klass volatility 
#########################

def Garman_Klass_volat(high, low, Open):
    volat_GK = np.sqrt((1/len(high)) * np.sum( 0.511* (np.log(high/low) **2) - 
            (0.019*np.log(low/Open) * np.log(low*high/Open)) - 
            (2*np.log(high/Open) * np.log(low/Open)))) 
    
    ann_volat_GK = np.sqrt(365) * volat_GK
    return ann_volat_GK
    
Garman_Klass_volat(tab1_df['high'], tab1_df['low'], tab1_df['open'])


#######################################################
# Simple Moving Average
#######################################################

def MA (price,n):
    
    MA = []
    MA = (price.rolling(window=n).mean())
    return MA 


Mov_Av_20days = MA (close,20)



tab4 = {'Close': pd.Series(close),'Mov20' : pd.Series(Mov_Av_20days)}
tab4_df = pd.DataFrame(tab4, columns=['Mov20', 'Ema20'])

#plot 2010 -2018
plt.close('all')
fig, ax =plt.subplots(1)
ax.plot( tab4_df['Mov20'], 'b', lw = 1.5, label = 'MovAv 20')
ax.plot(tab1_df['close'], 'g', lw = 1.5, label = 'BTC price')
plt.title('BTCUSD moving average review')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('price ($) ')
plt.xlabel('year')
plt.show()

#Focus 2018 previous figure 
closeFocus = close[1935:]
Mov_Av_20daysFocus = MA(close[1935:],20)


tab4_bis= {'close' : pd.Series(closeFocus),'Mov20' : pd.Series(Mov_Av_20daysFocus)}
tab4_bis_df = pd.DataFrame(tab4_bis, columns=['close', 'Mov20'])


plt.close('all')
fig, ax =plt.subplots(1)
ax.plot( tab4_bis_df['Mov20'], 'b', lw = 1.5, label = 'MovAv 20')

ax.plot(tab4_bis_df['close'], 'g', lw = 1.5, label = 'BTC price')
plt.title('BTCUSD moving average review')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('price ($) ')
plt.xlabel('year')
plt.show()




#######################################################
#Bollinger Bands
#######################################################

MovingAverage = tab1_df['close'].rolling(window=20).mean() #here we take 20 days for the 20 days MA due to the fact that the market is open 24/7 
STD = tab1_df['close'].rolling(window=20).std()
UpperBand = MovingAverage + (STD *2)
DownBand = MovingAverage - (STD *2)

plt.close('all')
fig, ax =plt.subplots(1)
ax.plot( tab1_df['close'], 'r', lw = 1.5, label = 'price')
ax.plot( MovingAverage , 'b', lw = 1.5, label = 'MA20')
ax.plot( UpperBand, 'y', lw = 1.5, label = 'UPband')
ax.plot( DownBand, 'g', lw = 1.5, label = 'DOWNband')
plt.title('Observed USDT/BTC prices 2010 - 2018 (1 day)')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('prices ($)')
plt.xlabel('number of days')
plt.show()



#It's not optimal we don't really see anything so we make a focus on 2018 

tab2 = {'close' : pd.Series(close), 'Moving_Average': pd.Series(MovingAverage), 'UpperBand' : pd.Series(UpperBand),
        'DownBand' : pd.Series(DownBand), 'STD' : pd.Series(STD) }                        

tab2_df = pd.DataFrame(tab2, columns=['close', 'Moving_Average', 'UpperBand', 'DownBand', 'STD'])

tab2_df = tab2_df.reset_index(drop=False)
tab2_df =tab2_df[1935:]
tab2_df.index = tab2_df['Date']
del tab2_df['Date']

plt.close('all')
fig, ax =plt.subplots(1)
ax.plot( tab2_df['close'], 'r', lw = 1.5, label = 'price')
ax.plot( tab2_df['Moving_Average'] , 'b', lw = 1.5, label = 'MA20')
ax.plot( tab2_df['UpperBand'], 'olive', lw = 1.5, label = 'UPband')
ax.plot( tab2_df['DownBand'], 'grey', lw = 1.5, label = 'DOWNband')
x_axis = tab2_df.index.get_level_values(0)
ax.fill_between(x_axis, tab2_df['UpperBand'], tab2_df['DownBand'], color='lightblue')
plt.title('Observed USDT/BTC prices 2018 (1 day)')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('prices ($)')
plt.xlabel('number of days')
plt.show()



#######################################################
#Commodity Channel Index
#######################################################


CCI_mean = (tab1_df['close'] + tab1_df['low'] + tab1_df['high'])/3
CCI_mov_mean = tab1_df['close'].rolling(window=14).mean()
CCI_std = tab1_df['close'].rolling(window=14).std()


tab3 = {'close': pd.Series(close),'CCI_mean' : pd.Series(CCI_mean), 'CCI_mov_mean' : pd.Series(CCI_mov_mean), 'CCI_mov_std' : pd.Series(CCI_std)}
tab3_df = pd.DataFrame(tab3, columns=['close','CCI_mean','CCI_mov_mean','CCI_mov_std'])


def CCI(close, mov_mean, mov_std): 
   CCI = (close - mov_mean)/(mov_std * 0.015)
   return CCI

CCI_index = CCI(tab1_df['close'],tab3_df['CCI_mov_mean'],tab3_df['CCI_mov_std'])

tab3_df['CCI_Index'] = CCI_index



CONSTANTE1 = 100
CONSTANTE2 = -100


plt.close('all')
fig, ax =plt.subplots(1)
ax.plot( tab3_df['CCI_Index'], 'silver', lw = 1.5, label = 'price')
plt.axhline(y=100, color='b', linestyle=':')
plt.axhline(y=-100, color='b', linestyle=':')
plt.title('CCI Index USDT/BTC 2010 - 2018 (1 day)')
plt.legend(loc=0)
plt.grid(True)
plt.ylabel('CCI value ')
plt.xlabel('number of days')
plt.show()





































 



































  
    
    
    
   
