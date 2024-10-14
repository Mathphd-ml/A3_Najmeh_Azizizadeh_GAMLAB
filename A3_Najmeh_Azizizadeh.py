"""
Created on Fri Oct  4 23:51:38 2024

@author: Najmeh Azizizadeh


Salam Ostad
darsadhay jadid ro neveshtam.
kheili run gereftam. Inha behtarin boodan, vali hanooz be deghat 90% nareside.
mishe lotfan negahi befarmaeed.
************
APM:
dar ghesmate SVR man yek raveshe jadid zadam ( normal kardane data ha) , hala run konid va agar darsadeton kamtar az -0.1 bod
paksazi konid va moratab benevisid va dar enteha ghesmate report bezarid va kamel begid data chi boode va che model haee netekhab kardid harkodom darsade deghat cheghadr
bode va kodom behtar bode va sabt konid baraye reporte final (payani)

"""

#-----------Import Libs----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#-----------Import Data----------------------
data=fetch_california_housing()



#-----------Step1 : X and Y ----------------------

x=data.data
y=data.target

#---------- Step2: kfold -------------------------

kf= KFold(n_splits=5,shuffle=True,random_state=42)

#---------- Step3: Model 1 -------------------------

model=LinearRegression()

my_params= {'fit_intercept':[True,False]}

gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     #np.float64(-0.3175876329794739)
gs.best_params_    #{'fit_intercept': True}
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 2 -------------------------

model= KNeighborsRegressor()

my_params= { 'n_neighbors':[3,5,7,8,9],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'],
           'weights':['uniform','distance']}


gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     #np.float64(-0.21067448749397003)
gs.best_params_    #{'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 3 -------------------------

model=DecisionTreeRegressor(random_state=42)

my_params={ 'max_depth':[1,2,3,24,21,17,32,10],
           'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           'splitter':['best','random'],'max_leaf_nodes':[798,810,805]}


gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
X_scaled = scaler.fit_transform(x)
gs.fit(X_scaled,y)
gs.best_score_     #np.float64(-0.2202894825724071)
gs.best_params_   
'''
 {'criterion': 'absolute_error',
 'max_depth': 24,
 'max_leaf_nodes': 805,
 'splitter': 'best'}
 '''
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 4 -------------------------

model=RandomForestRegressor(random_state=42)

my_params={ 'n_estimators':[10,20,30,70,100],
           'max_features':[1,2,3,4,5,6,7],
           'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           'max_depth':[20,30,25]}



gs=GridSearchCV(model,my_params,cv=kf,n_jobs=-1,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(X_scaled,y)
gs.best_score_   #   np.float64(-0.17956506039641587)
gs.best_params_    
'''
{'criterion': 'poisson',
 'max_depth': 20,
 'max_features': 7,
 'n_estimators': 100}
'''
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 5 -------------------------

model=SVR()
my_params={'kernel':['poly','rbf','linear'], 'epsilon':[0.01,0.00001], 'tol':[0.000001],
           'C':[0.001,0.01],'degree':[2,3,7],'gamma':['scale','auto'],
           'cache_size':[100,200],'coef0':[2,1]}

gs=GridSearchCV(model,my_params,cv=kf,n_jobs=-1,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
#******
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x) 


gs.fit(x_scaled,y)
gs.best_score_     #np.float64(-0.2346791171309992)
gs.best_params_
'''
{'C': 0.01,
 'cache_size': 100,
 'coef0': 2,
 'degree': 7,
 'epsilon': 0.01,
 'gamma': 'scale',
 'kernel': 'poly',
 'tol': 1e-06}
''' 
cv=gs.cv_results_


