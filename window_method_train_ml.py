import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

tune_knn=False


data=pd.read_csv(Path("ml_data/statistics_set.csv"),index_col=0)
y=data["gains"].values
y[y<=0]=0
y[y>0]=1
X=data.drop(["gains"],axis=1).values
    
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.4)

scalerX=preprocessing.StandardScaler().fit(X_train)
X_train_s=scalerX.transform(X_train)
X_test_s=scalerX.transform(X_test)

#Logistic Regression (classification)
logm=LogisticRegression().fit(X_train,y_train)
logm_train=logm.score(X_train,y_train)
logm_test=logm.score(X_test,y_test)
print("Logistic Regression")
print("Training Accuracy")
print(logm_train)
print("Testing Accuracy")
print(logm_test)

#K nearest neighbors
#   hyperperamater optimization
if tune_knn==True:
    canidate_n=list(range(131,171,2))
    n_scores=[]
    for n in canidate_n:
        tempscore=[]
        for _ in range (45):
            X_train_temp,X_test_temp,y_train_temp,y_test_temp=train_test_split(X,y,test_size=.4)
            tempmodel=KNeighborsClassifier(n_neighbors=n).fit(X_train_temp,y_train_temp)
            tempscore.append(tempmodel.score(X_test_temp,y_test_temp))
        n_scores.append(np.mean(tempscore))
    
    nidx_best=np.argmax(n_scores)
    n_best=canidate_n[nidx_best]
else:
    n_best=147

#   final run        
knnm=KNeighborsClassifier(n_neighbors=n_best).fit(X_train,y_train)
knnm_train=knnm.score(X_train,y_train)
knnm_test=knnm.score(X_test,y_test)

print("K Nearest Neighbor")
print("N")
print(n_best)
print("Training Accuracy")
print(knnm_train)
print("Testing Accuracy")
print(knnm_test)


