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

#nPurchases = df.groupby('Item').size()
#fTable = pd.Series.sort_index(pd.Series.value_counts(nPurchases))
#print(fTable)
#print()

#nItems = df.groupby('Customer').size()
#fTable = pd.Series.sort_index(pd.Series.value_counts(nItems))
#print(fTable)
#print()

itemSets = apriori(ItemIndicator, min_support = 0.007625826, use_colnames = True)
print(itemSets)

kMax = 0

for i in range(0, 524):
	kItemset = itemSets.iloc[i].itemsets
	k = len(kItemset)
	
	if k > kMax:
		kMax = k
		
print('Max K: ', kMax)
print()

rules = association_rules(itemSets, metric = 'confidence', min_threshold = 0.01)
print(rules)

rules2 = association_rules(itemSets, metric = 'confidence', min_threshold = 0.6)
print('Rules with confidence above 60%\n')
print(rules2)
print()

fig, ax = plt.subplots(figsize = (10, 10))
plt.scatter(rules['confidence'], rules['support'], s = rules['lift'])
bins = plt.hexbin(rules['confidence'], rules['support'], C = rules['lift'], bins = 20, gridsize = 50)
bar = fig.colorbar(bins, ax = ax)
bar.set_label('Lift')
plt.grid(True)
plt.xlabel('Confidence')
plt.ylabel('Support')
plt.show()