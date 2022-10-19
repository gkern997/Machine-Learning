# This program trains a series of multi-layer perceptron neural networks to predict the frequency of homeowner insurance claims
# The training dataset "Homeowner_Claim_History.csv" contains the claim history of 27,513 policies as well as 7 categorical predictors
# Namely:

# f_primary_age_tier: The age tier of the policyholder (< 21, 21 - 27, 28 - 37, 38 - 60, 60+)
# f_primary_gender: The gender of the policyholder (Male, Female)
# f_marital: The marital status of the policyholder (Not married, Married, Un-Married)
# f_residence_location: The zoning of the insured property (Urban, Suburban, Rural)
# f_fire_alarm_type: Type of fire alarm present on the insured property (None, Standalone, Alarm Service)
# f_mile_fire_station: Distance from the insured property to the nearest fire station (< 1mi, 1 - 5mi, 6 - 10mi, 10+mi)
# f_aoi_tier: Value of the insured property (< 100K, 100 - 350K, 351 - 600K, 601K - 1M, 1M+)
  
# Author: Grayson Kern

import numpy as np
import pandas as pd
import itertools
import statistics as stats
import matplotlib.pyplot as plt
import sklearn.neural_network as nn
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
import time
import warnings
warnings.filterwarnings("ignore")

inputData = pd.read_csv('Homeowner_Claim_History.csv', usecols = ['exposure', 'num_claims', 'f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier'], delimiter = ',')
inputData['Frequency'] = inputData['num_claims'] / inputData['exposure']

catNames = ['f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']
yName = 'Frequency'
scaler = MinMaxScaler()

trainData = inputData[catNames + [yName]].dropna().reset_index(drop = True) # Preparing the training data
nSamples = trainData.shape[0]



for pred in catNames:
	u = trainData[pred].astype('category').copy()
	u_freq = u.value_counts(ascending = True)
	trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

X = pd.get_dummies(trainData[catNames].astype('category')) # Assign dummy values to the predictor categories
Y = trainData[yName]

res = pd.DataFrame()

# Network parameters to iterate over 
actFunc = ['identity', 'tanh'] # Activation function list
nLayers = range(1, 11, 1) # Amount of layers
nHidden = range(1, 6, 1) # Neurons per layer 

comboList = itertools.product(actFunc, nLayers, nHidden)
relErrNum = 0
relErrDen = 0
YMean = stats.mean(trainData[yName])

for i in range(len(trainData[yName])):
	diffSq = np.power((trainData[yName][i] - YMean), 2)
	relErrNum = relErrNum + diffSq

# Train and fit a network for each combination of activation function, layer count, and neurons per layer
# Note: this takes quite some time (10+ min on my high end desktop)

for combo in comboList:
	startTime = time.time()
	actFunc = combo[0]
	nLayers = combo[1]
	nHidden = combo[2]
	
	nnObj = nn.MLPRegressor(hidden_layer_sizes = (nHidden,)* nLayers, activation = actFunc, verbose = False, max_iter = 10000, random_state = 31010)
	fit = nnObj.fit(X, Y)
	yPred = nnObj.predict(X)
	yAct = trainData[yName]
	
	for i in range(len(yPred)):
		diffSq = np.power((trainData[yName][i] - yPred[i]), 2) 
		relErrDen = relErrDen + diffSq
	
	relErr = (relErrNum / relErrDen) # Relative Error
	
	print('Function: ', combo[0])
	print('nLayers: ', combo[1])
	print('nHidden: ', combo[2])
	
	yRes = yAct - yPred
	meanSqErr = np.mean(np.power(yRes, 2))
	rase = np.sqrt(meanSqErr) # Root average squared error
	RSquared = metrics.r2_score(yAct, yPred) # Pearson correlation
	elapsedTime = time.time() - startTime # Elapsed time of each iteration
	nIterations = nnObj.n_iter_ # Number of iterations
	bestLoss = nnObj.best_loss_ # Best loss value
		
	print('RASE: ', rase)
	print('Relative Error: ', relErr)
	print('Pearson: ', RSquared)
	print()
	
	res = res.append([[actFunc, nLayers, nHidden, nIterations, bestLoss, rase, relErr, RSquared, elapsedTime]], ignore_index = True)
	
res.columns = ['Activation Function', 'nLayers', 'nHidden', 'Iterations', 'Best Loss', 'RASE', 'Relative Error', 'Pearson Correlation', 'Elapsed Time']
pd.set_option('display.max_rows', None, 'display.max_columns', None)
print(res)
print()
res.to_csv(path_or_buf = 'res.csv', sep = ',')

# Locate the optimal architecture
optidx = res['RASE'].idxmin()
optrow = res.iloc[optidx]
actFunc = optrow['Activation Function']
nLayers = optrow['nLayers']
nHidden = optrow['nHidden']
R2 = optrow['Pearson Correlation']

print('============')
print('Best Network')
print('============')
print(optidx)
print(optrow)






