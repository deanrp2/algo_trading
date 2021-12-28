import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import NuSVR

data=pd.read_csv(Path("ml_data/statistics_set.csv"),index_col=0)
y=data["gains"].values.reshape(-1,1)
X=data.drop(["gains"],axis=1).values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)

scalery=preprocessing.StandardScaler().fit(y_train)
scalerX=preprocessing.StandardScaler().fit(X_train)

y_train_s=scalery.transform(y_train)
y_test_s=scalery.transform(y_test)
X_train_s=scalerX.transform(X_train)
X_test_s=scalerX.transform(X_test)

#Linear Model
linear_model=LinearRegression().fit(X_train_s,y_train_s)
linear_model_r_train=linear_model.score(X_train_s,y_train_s)
linear_model_r_test=linear_model.score(X_test_s,y_test_s)
print("Linear Model train",linear_model_r_train)
print("Linear Model test",linear_model_r_test)

#Random Forest Regression
rfr_model=RandomForestRegressor(max_depth=6).fit(X_train_s,y_train_s.ravel())
rfr_model_r_train=rfr_model.score(X_train_s,y_train_s.ravel())
rfr_model_r_test=rfr_model.score(X_test_s,y_test_s.ravel())
print("Random Forest Regression train",rfr_model_r_train)
print("Random Forest Regression test",rfr_model_r_test)

#Bayesian Ridge Regression
bay_model=BayesianRidge().fit(X_train_s,y_train_s.ravel())
bay_model_r_train=bay_model.score(X_train_s,y_train_s.ravel())
bay_model_r_test=bay_model.score(X_test_s,y_test_s.ravel())
print("Bayesian Ridge Regression train",bay_model_r_train)
print("Bayesian Ridge Regression test",bay_model_r_test)

#Gaussian Process Regrsssion
#gpr_model=GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()).fit(X_train_s,y_train_s.ravel())
#gpr_model_r_train=gpr_model.score(X_train_s,y_train_s.ravel())
#gpr_model_r_test=gpr_model.score(X_test_s,y_test_s.ravel())
#print("Gaussian Process Regression train",gpr_model_r_train)
#print("Gaussian Process Regression test",gpr_model_r_test)

#Support Vector Machine
svm_model = NuSVR(C=1.0,nu=0.1).fit(X_train_s,y_train_s.ravel())
svm_model_r_train=svm_model.score(X_train_s,y_train_s.ravel())
svm_model_r_test=svm_model.score(X_test_s,y_test_s.ravel())
print("Support Vector Machine train",svm_model_r_train)
print("Support Vector Machine test",svm_model_r_test)





