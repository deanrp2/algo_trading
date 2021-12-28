import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random
import time
from pathlib import Path
import datetime

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)

def Convert(string):
    li = list(string.split("\n"))
    return li

def getbad(good,bad):
    totalrows=good.shape[0]
    missing=totalrows-bad.shape[0]

    num_nans=np.zeros(bad.shape[1])
    for i,c in enumerate(bad.columns):
        num_nans[i]=np.count_nonzero(~np.isnan(bad[c]))
        if np.isnan(good[c].iloc[0]):
            num_nans[i]= 0
    return (totalrows-num_nans)/totalrows

def gsdata(s,fill=True,startdate=None,enddate=None,countbads=False):
    if isinstance(s, str):
        s = [s] #this just places s as a single-element list, makes it so if we only want one stock we do not need to put it in a list

    if len(s) == 1: #check if length of s is equal to 1
        col_suffix = s[0] #we only need this to be equal to the ticker if we are only getting one stock so we can fix the columns of the table that yahoo gives us
    else:
        col_suffix = "" #if s is a list then we will include the ticker when we collapse that multiindex thing

    ss=" ".join(s)
    rawdata = yf.download(ss,period = "1d", interval = "1m",start=startdate, end=enddate)
    rawdata.columns=[''.join(col).strip() + col_suffix for col in rawdata.columns.values]
    filldata = rawdata.asfreq("1Min", "pad").resample("1Min").asfreq().ffill() #create a different variable that has the values filled

    if fill==True:
        returndata = filldata
    elif fill==False:
        returndata = rawdata

    if countbads==False:
        return returndata
    elif countbads==True:
        return returndata, getbad(filldata, rawdata)


start= datetime.datetime(2021,12,16)
end= datetime.datetime(2021,12,22)

destination= Path("stock_data")

with open(Path("tickers.txt"), "r") as ff:
    s=ff.read() 

tickers =  Convert(s)

for date in daterange(start, end):
    print(date)
    if date.weekday() not in set([5,6]):
        hist,badfrac=gsdata(tickers,startdate=date,enddate=date+datetime.timedelta(days=1),countbads=True)#hist is dataframe of stock data badfrac is list of fractions of nan in each column
        goodcols=badfrac<.1
        goodhists=hist.iloc[:,goodcols]
        goodhists.to_csv(destination/(str(date)[0:10]))
    else:
        print("this a weekend")



