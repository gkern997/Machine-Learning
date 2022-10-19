# This program uses a categorical naive bayes calssifier to predict the purchase probabilities of 3 types of insurance policy based on customer demographics
# The training data "Purchase_likelihood.csv" contains 665,249 observations across 97,009 unique Allstate customers 
# Author: Grayson Kern

import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn import naive_bayes as nb

# Target Variable = Insurance = {0, 1, 2}
# Predictor 1 = group size = {1, 2, 3, 4}
# Predictor 2 = homeowner = {0, 1}
# Predictor 3 = maritial status = {0, 1}

def rowWithcolumn(rowVar, columnVar, show): # Generates frequency and fraction tables 

	countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
	print("Frequency Table: \n", countTable)
	print()
	
	if(show == 'ROW' or show == 'BOTH'):
		rowFrac = countTable.div(countTable.sum(1), axis = 'index')
		print("Row Fraction Table: \n", rowFrac)
		print()
		
	elif(show == 'COLUMN' or show == 'BOTH'):
		columnFrac = countTable.div(countTable.sum(0), axis = 'columns')
		print("Column Fraction Table: \n", columnFrac)
		print()
		
	else:
	
		return

trainData = pd.read_csv('Purchase_Likelihood.csv', delimiter = ',')

subTrainData = trainData[['group_size', 'homeowner', 'married_couple', 'insurance']].dropna()

catInsurance = subTrainData['insurance'].unique()
catMarriedCouple = subTrainData['married_couple'].unique()
catHomeowner = subTrainData['homeowner'].unique()
catGroupSize = subTrainData['group_size'].unique()

# Display the frequency and fraction tables of each insurance policy class by each demographic predictor 
rowWithcolumn(rowVar = subTrainData['insurance'], columnVar = subTrainData['married_couple'], show = 'ROW')
rowWithcolumn(rowVar = subTrainData['insurance'], columnVar = subTrainData['group_size'], show = 'ROW')
rowWithcolumn(rowVar = subTrainData['insurance'], columnVar = subTrainData['homeowner'], show = 'ROW')

subTrainData = subTrainData.astype('category')
xTrain = subTrainData[['group_size', 'homeowner', 'married_couple']].to_numpy() # Process the training data 
yTrain = subTrainData['insurance'].to_numpy()

NB = nb.CategoricalNB(alpha = 1.0e-10) # Create and train a naive bayes model based on the training data
model = NB.fit(xTrain, yTrain)

# Manually enumerate every combination of customer demographics
x1 = [['1', '0', '0']]
x2 = [['1', '0', '1']]
x3 = [['1', '1', '0']]
x4 = [['1', '1', '1']]
x5 = [['2', '0', '0']]
x6 = [['2', '0', '1']]
x7 = [['2', '1', '0']]
x8 = [['2', '1', '1']]
x9 = [['3', '0', '0']]
x10 = [['3', '0', '1']]
x11 = [['3', '1', '0']]
x12 = [['3', '1', '1']]
x13 = [['4', '0', '0']]
x14 = [['4', '0', '1']]
x15 = [['4', '1', '0']]
x16 = [['4', '1', '1']]

# Use model fit to predict the probability of each customer demographic purchasing each insurance policy type 
y1 = model.predict_proba(x1)
y2 = model.predict_proba(x2)
y3 = model.predict_proba(x3)
y4 = model.predict_proba(x4)
y5 = model.predict_proba(x5)
y6 = model.predict_proba(x6)
y7 = model.predict_proba(x7)
y8 = model.predict_proba(x8)
y9 = model.predict_proba(x9)
y10 = model.predict_proba(x10)
y11 = model.predict_proba(x11)
y12 = model.predict_proba(x12)
y13 = model.predict_proba(x13)
y14 = model.predict_proba(x14)
y15 = model.predict_proba(x15)
y16 = model.predict_proba(x16)

print("Customer Type (group size, homeowner, marital status) \t", x1)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y1)
print()

print("Customer Type (group size, homeowner, marital status) \t", x2)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y2)
print()

print("Customer Type (group size, homeowner, marital status) \t", x3)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y3)
print()

print("Customer Type (group size, homeowner, marital status) \t", x4)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y4)
print()

print("Customer Type (group size, homeowner, marital status) \t", x5)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y5)
print()

print("Customer Type (group size, homeowner, marital status) \t", x6)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y6)
print()

print("Customer Type (group size, homeowner, marital status) \t", x7)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y7)
print()

print("Customer Type (group size, homeowner, marital status) \t", x8)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y8)
print()

print("Customer Type (group size, homeowner, marital status) \t", x9)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y9)
print()

print("Customer Type (group size, homeowner, marital status) \t", x10)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y10)
print()

print("Customer Type (group size, homeowner, marital status) \t", x11)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y11)
print()

print("Customer Type (group size, homeowner, marital status) \t", x12)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y12)
print()

print("Customer Type (group size, homeowner, marital status) \t", x13)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y13)
print()

print("Customer Type (group size, homeowner, marital status) \t", x14)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y14)
print()

print("Customer Type (group size, homeowner, marital status) \t", x15)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y15)
print()

print("Customer Type (group size, homeowner, marital status) \t", x16)
print("Predicted Purchase prob (class 0, class 1, class 2) \t", y16)
print()




