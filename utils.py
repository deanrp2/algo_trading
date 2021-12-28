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
        
        temp_price_columns=[]
        for c in temp_price.columns:
            l=len(pricepoint)
            temp_price_columns.append(c[l:]+date)
        temp_price.columns=temp_price_columns

        temp_volume_columns=[]
        for d in temp_volume.columns:
            l=len("Volume")
            temp_volume_columns.append(d[l:]+date)
        temp_volume.columns=temp_volume_columns

        concat_price.append(temp_price)
        concat_volume.append(temp_volume)

    price=pd.concat(concat_price,axis=1)
    volume=pd.concat(concat_volume,axis=1)
    return price,volume


