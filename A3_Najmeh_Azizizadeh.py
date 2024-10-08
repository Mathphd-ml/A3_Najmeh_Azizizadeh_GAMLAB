"""
Created on Fri Oct  4 23:51:38 2024

@author: Najmeh Azizizadeh


Bale Chashm Ostad.

---javab----
salam arz shdo, darsad haye erroreton balast yani deghateton nahayat 74 ina hast , decision tree, random forest va svr ba hypeparameter ha bazi konid brid]
dakhele documentation baghie hypeparameter haro ezafe koid t abetonid hadeaghal 90% deghat bgeirid

nokte---> MAPE --> vaghti mige -0.24 yani --> 0.24 yani 24 / 100 --> yani 24% khata (error)---> yani 100-24--> 76% deghat. shoma bayad ye adad hodode
-0.10 begirid yani --> 10% khata ya --> 100 - 10 = 90% deghat


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

my_params= { 'n_neighbors':[2,3,4,7,8,9],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }


gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     #np.float64(-0.4847991767838546)
gs.best_params_    #{'metric': 'manhattan', 'n_neighbors': 4}
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 3 -------------------------

model=DecisionTreeRegressor(random_state=42)

my_params={ 'max_depth':[1,2,3,4,5,7,8,10]}


gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     #np.float64(-0.24051815083527037)
gs.best_params_    #{'max_depth': 10}
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 4 -------------------------

model=RandomForestRegressor(random_state=42)

my_params={ 'n_estimators':[10,20,30,70],
           'max_features':[1,2,3,4,5,6,7],
           'max_depth':[2,3,4,5,7,8]}



gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_   #   np.float64(-0.23638255779254744)
gs.best_params_    #{'max_depth': 8, 'max_features': 7, 'n_estimators': 70}
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


