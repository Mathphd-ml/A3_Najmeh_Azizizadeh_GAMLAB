"""
Created on Fri Oct  4 23:51:38 2024

@author: Najmeh Azizizadeh


Salam Ostad
SVR kheili kond shode, man faghat gamma ro ezafeh kardam. 48 saat run shod vali javab nadad.
Random forest ham max_depth bozorgtar zadam vali baad az 30 saat run javab nadad, stop kardam.

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
gs.best_score_     #np.float64(-0.0.478347758446281)
gs.best_params_    #{'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 3 -------------------------

model=DecisionTreeRegressor(random_state=42)

my_params={ 'max_depth':[1,2,3,24,21,17,22,10],
           'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           'splitter':['best','random'],'max_leaf_nodes':[798,810,805]}


gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     #np.float64(-0.22072829758907314)
gs.best_params_   
'''
 {'criterion': 'absolute_error',
 'max_depth': 21,
 'max_leaf_nodes': 805,
 'splitter': 'random'}
 '''
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 4 -------------------------

model=RandomForestRegressor(random_state=42)

my_params={ 'n_estimators':[10,20,30,70,90],
           'max_features':[1,2,3,4,5,6,7],
           'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           'max_depth':[2,3,4,25]}



gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_   #   np.float64(-0.1818179719877024)
gs.best_params_    #{'max_depth': 25, 'max_features': 4, 'n_estimators': 90}
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 5 -------------------------

model=SVR()
my_params={'kernel':['poly','rbf','linear'],
           'C':[0.001,0.01,1]}


gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     #np.float64(-0.29294216082111857)
gs.best_params_    #{'C': 0.01, 'kernel': 'linear'}   
cv=gs.cv_results_


