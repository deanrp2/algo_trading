import numpy as np
import pandas as pd
from pathlib import Path



def pick_type(stockdata,pricepoint):
    dcolumns=[f for f in stockdata.columns if pricepoint in f]
    return stockdata[dcolumns]


def pick_volume(stockdata):
    dcolumns=[f for f in stockdata.columns if "Volume" in f]
    return stockdata[dcolumns]



def read_datafiles(dates,pricepoint):
    concat_price=[]
    concat_volume=[]
    for date in dates:
        temp= pd.read_csv(Path("stock_data")/Path(date))
        temp_price= pick_type(temp, pricepoint)
        temp_volume= pick_volume(temp)
        
        print (temp_volume)

read_datafiles(["2021-12-16","2021-12-17"],"Open")

