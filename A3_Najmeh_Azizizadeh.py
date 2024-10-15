"""
Created on Fri Oct  4 23:51:38 2024

@author: Najmeh Azizizadeh
#---------------------------
'''
Salam Ostad
Natayej nahaeei ra vared kardam.
faghat bebakhshid dar ravesh SVR vaghti c=1 migozashtam, kheili tool mikeshid(5 saat bishtar ke dighe motevaghefash mikardam) hata ba scale kardan. 
baraye hamin shayad deghat model man khoob nashodeh.
Rasm ham nafahmidam chetor har 5 ravesh ra mitoonam dar yek nemoodar biyavaram. ye dastoor koli neveshtam.

Dar payan ham az shoma besyar sepasgozaram, babate elmi ke behem yad dadid, hich vaght fekr nimikardam betoonam Python ra yad begiram va azash khosham biyad.
shoma aali tadris mikonid, Mamnoonam.
Movafaghiyathayetan roozafzoon.
'''
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

my_params={ 'max_depth':[24,35,17,12,21],
           'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           'max_features':[1,2,3,4,5,6,7],
           'splitter':['best','random'],'max_leaf_nodes':[700,500,200,605]}


gs=GridSearchCV(model,my_params,cv=kf,n_jobs=-1,scoring='neg_mean_absolute_percentage_error')

#-----------Step4:  fit -------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
X_scaled = scaler.fit_transform(x)
gs.fit(X_scaled,y)
gs.best_score_     #np.float64(-0.2114834767005333)
gs.best_params_   
'''
 {'criterion': 'absolute_error',
 'max_depth': 17,
 'max_features': 7,
 'max_leaf_nodes': 500,
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
#------------------------------------------------------------------------------------
#--------- Plot -----------------------------------------------

plt.scatter(X_scaled[:,4],y)
plt.xlabel('AveBedrms')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
#------------------------------------------------------------------------
'''
Report:
  دیتا مربوط به 20640 خانه در کالیفرنبا است که 8 مولفه از آنها
    رو بررسی کرده و در قسمت تارگت(هدف) قیمت خانه را آورده.
x.data:  ['MedInc',
   'HouseAge',
   'AveRooms',
   'AveBedrms',
   'Population',
   'AveOccup',
   'Latitude',
   'Longitude']  
y.data: 'MedHouseVal'
میخواهیم رابطه بین این مولفه های خانه ها را با قیمتشون در بیاوریم.
با 5 روش این کار را انجام میدهیم و میخواهیم ببینیم که کدوم مدل بهترین پیش بینی را میدهد.
(یعنی اگر خانه جدیدی را به مدل بدهیم بتواند قیمت را پیش بینی کنه)
در ضمن چون قسمت هدف داده، مشخص و به صورت اعداد اعشاری میباشد(قیمت)
 از شاخه Supervised Regression استفاده میکنیم.
همچنین برای مدل های 2 تا 5 داده ها را نرمال کردیم.

  مدل اول: رگرسیون خطی (Linear Regression) خطای مدل تقریباً 31% و دقت آن حدود 69% میباشد.
  مدل دوم: نزدیکترین همسایگی (K-Nearest Neighbors) با خطای 21% و دقت تقریبی 79% است.
  مدل سوم: درخت تصمیم(DecisionTree) با خطای حدود 21% و دقت 79%
  مدل چهارم: جنگل تصادفی (Random Forest) با خطای تقریبی 17% و دقت تقریبا 83%
  مدل پنجم: ماشین بردار پشتیبان(Support Vector Machine) خطای این روش تقریبا 23% و دقت 77% میباشد.
بهترین مدل Random Forest است که دقت بهتر و بیشتری نسبت به سایر مدل ها دارد. 
'''
