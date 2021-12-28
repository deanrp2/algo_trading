import numpy as np
import pandas as pd
from pathlib import Path
import utils
import matplotlib.pyplot as plt
import scipy.stats as sp

dates=["2021-12-13","2021-12-16","2021-12-17","2021-12-20","2021-12-21","2021-12-22"]
training_cutoff=150 #beginning of prediciton window
trading_time=200 #end of prediction window, trades must be made in prediction window
cutoff_zone=15 #buffer between buy and trading_time
price, volume= utils.read_datafiles(dates,"Open")

def buysell_choice(prediction):
    buy=None
    sell=None
    if prediction.argmin()<prediction.size-cutoff_zone:
        buy=prediction.argmin()
        sell=prediction[buy:].argmax()+buy
        return (buy+training_cutoff, sell+training_cutoff)
    else:
        return (buy,sell)

gains=[] #gain from implementing trading strategy
tot_volume=[] #total volume in training period
frac_change=[] #price at trainging cutoff/open
rsquared=[] #r squared from linear modle fit to traing data
trend=[] #linear coeffecient from model fit to training data

for c in price.columns:
    price_prediction=utils.fourier_extrapolation(price[c][:training_cutoff],trading_time-training_cutoff)
    buy,sell=buysell_choice(price_prediction[training_cutoff:trading_time])
    if not buy is None  :
        gains.append(price[c][sell]/price[c][buy]-1)
    else:
        gains.append(0)
    tot_volume.append(volume[c][:training_cutoff].sum())
    frac_change.append(price[c][training_cutoff]/price[c][0])
    scaled_temp=price[c]/price[c][0]
    slope, intercept, r_value, p_value, std_err = sp.linregress(np.arange(training_cutoff), scaled_temp[:training_cutoff])
    rsquared.append(r_value**2)
    trend.append(slope)

ml_data=pd.DataFrame({"gains":gains,"tot_volume":tot_volume,"frac_change":frac_change,"rsquared":rsquared,"trend":trend})

ml_data.to_csv(Path("ml_data/statistics_set.csv"))
