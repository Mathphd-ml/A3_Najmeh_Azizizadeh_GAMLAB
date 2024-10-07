"""
Created on Fri Oct  4 23:51:38 2024

@author: Najmeh Azizizadeh

JAVAB DADE SHOD ( BAD AZ KHOONDAN INJARO PAK KONID)

JAVAB ---> KHATE 110
khate 145









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

my_params={ 'n_estimator':[10,20,30,40,50,100],
           'max_features':[1,2,3,4,5,6,7],
           'max_depth':[2,3,4,5,7,8]}

'''
Bebakhshid inja eshkal daram. fit ro anjam nemide. fekr konam params eshtebah neveshtam.
'''

'''
hamishe say konid b documentation berid
documentatione randomforest:
https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html

n_estimator nist
n_estimators
s tahesh ro nazashtid 



'''



gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     
gs.best_params_    
cv=gs.cv_results_

#------------------------------------------------------------------------------
#--------------- Step3: Model 5 -------------------------


model=SVR()
my_params={'kernel':['poly','rbf','linear'],
           'C':[0.001,0.01,1,10]}

'''
Bebakhshid inja fit ro anjam nemide. yani laptopam hang mikone. hich kari ro baad az fit anjam nemide, majbooram az spyder biyam biroon.
'''

'''
harmoghe k run nashod kh sade samte rast paeen hamonja ke result ro neshon mide (behehs migan console) oonja
ye morabaye ghermez hast chanbar bznid --> intrupt by user mishe yani ghat shode tavasote shoma (user)
yeki az dalayele run nashodan ine ke shayad tool mikeshe, choon fit mitone 10 daghighe ya 100 daghighe ya 2,3 rooz beshe ( vaghty ba big data ya big hypeparameters tarafim)
ama yeki dg az dalayel ine ke laptobe shoma C bishtar az 1 ro ejaze nmide pas my_params to 10 ro hazf konid ag nshod baz bgid

**yadeton nare bad az doros krdne hameye ina , hazf konid comment harye man ro 
moafagh bashid

'''
gs=GridSearchCV(model,my_params,cv=kf,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
gs.fit(x,y)
gs.best_score_     
gs.best_params_    
cv=gs.cv_results_


