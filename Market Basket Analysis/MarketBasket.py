# This program discovers association rules between items in the "Groceries.csv" file
# Said file contains market basket data for 9835 unique customers 
# Author: Grayson Kern



import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from IPython.display import display

df = pd.read_csv('Groceries.csv', delimiter = ',')
itemList = df.groupby(['Customer'])['Item'].apply(list).values.tolist()

te = TransactionEncoder()
teList = te.fit(itemList).transform(itemList)
ItemIndicator = pd.DataFrame(teList, columns = te.columns_)

itemSets = apriori(ItemIndicator, min_support = 0.007625826, use_colnames = True) # Only consider itemsets with support level >= 75
print('Itemsets with support >= 75\n')
print(itemSets)
print()


rules = association_rules(itemSets, metric = 'confidence', min_threshold = 0.01) # Discover association rules in the frequent itemsets
print('Association rules with confidence >= 1%\n')
print(rules)
print()

rules2 = association_rules(itemSets, metric = 'confidence', min_threshold = 0.6)
print('Rules with confidence above 60%\n')
print(rules2)
print()

# Generate a plot of confidence vs support for the discovered rules 
fig, ax = plt.subplots(figsize = (10, 10))
plt.scatter(rules['confidence'], rules['support'], s = rules['lift'])
bins = plt.hexbin(rules['confidence'], rules['support'], C = rules['lift'], bins = 20, gridsize = 50)
bar = fig.colorbar(bins, ax = ax)
bar.set_label('Lift')
plt.grid(True)
plt.xlabel('Confidence')
plt.ylabel('Support')
plt.show()