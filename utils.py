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


def fourier_extrapolation(x, n_predict):
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t         # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

