"""
Created on Fri Oct  4 23:51:38 2024

@author: Najmeh Azizizadeh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

data=fetch_california_housing()



x=data.data
y=data.target
