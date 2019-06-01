#%%
# Load necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.cluster import KMeans

#%%
# Read data and present

train = pd.read_csv('trainingData.csv')
valid = pd.read_csv('validationData.csv')
def unnest(df, col):
    unnested = (df.apply(lambda x: pd.Series(x[col]), axis=1)
                .stack()
                .reset_index(level=1, drop=True))
    unnested.name = col
    return df.drop(col, axis=1).join(unnested)

def to_bag_of_cards(df):
    df['ind'] = np.arange(df.shape[0]) + 1
    df_orig = df.copy()
    df['deck'] = df['deck'].apply(lambda d: d.split(';'))
    df = unnest(df, 'deck')
    df['value'] = 1
    df_bag = df.pivot(index='ind', columns='deck', values='value')
    df_bag[df_bag.isna()] = 0
    df_bag = df_bag.astype('int')
    return pd.concat([df_orig.set_index('ind'), df_bag], axis=1)
 # %%
train2 = to_bag_of_cards(train)
valid2 = to_bag_of_cards(valid)
train2.head()

#%%
# Sort data by number of games played

train2 = train2.sort_values('numof', ascending=False)
valid2 = valid2.sort_values('dist', ascending=False)
#%%
# Specify example model fitting function and R squared metric


def R2(x, y):
    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y)))

def fit_svr(data):
    svr = SVR(kernel='rbf', gamma=1.0/90, C=1.0, epsilon=0.02, shrinking=False)
    svr.fit(data, data['winRate'])
    return svr

sizes = (np.arange(10) + 6) * 100

#%%
train3=train2.iloc[:,4:]
kmeans = KMeans(n_clusters=100)
kmeans.fit(train3)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(train3)
#%%
train3_dist=kmeans.transform(train3)
train3['dist']=train3_dist.min(axis=1)
train3['prediction']=y_km
train3['ID']=train3.index
df = train3.groupby('prediction')['ID'].nunique()
#%%
# Fit and predict on models of various training sizes
fit_list = list(map(lambda size: fit_svr(train3.iloc[:size]), sizes))
pred_list = list(map(lambda fit: fit.predict(valid2.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                     fit_list))
#%%
# Calculate R squared scores

r2 = list(map(lambda p: R2(p, valid2['winRate']), pred_list))
r2
#%%
_ = plt.plot(sizes, r2)
#%%
np.mean(r2)
#%%`
# Save hyperparameteres and selected indices in submission format

train3 = train3.sort_values('dist', ascending=True)
open('example_sub_python.txt', 'w').close()

with open('example_sub_python.txt', 'a') as f:
    for size in sizes:
        ind_text = ','.join(list(map(str, train3.index.values[:size])))
        text = ';'.join(['0.02', '1.0', str(1.0 / 90), ind_text])
        f.write(text + '\n')
#%%
def getIndexesFromCluster(data, clasterID,numberOfElements,list):
    pointsFromCluster=data[data['prediction'] == clasterID]
    pointsFromCluster=pointsFromCluster.sort_values(by='dist', axis=0, ascending=True)
    pointsFromCluster=pointsFromCluster[:numberOfElements]
    l=pointsFromCluster.index.tolist()
    for i in l:
        list.append(i)


def getIndex(data,numberOfClusters, numberOfElements, lista):
    num=int(numberOfElements/numberOfClusters)
    for i in range(numberOfClusters):
        getIndexesFromCluster(data, i, num, lista)

#%%

open('clusters.txt', 'w').close()

with open('clusters.txt', 'a') as f:
    for size in sizes:
        l = list()
        getIndex(train3, 100, size, l)
        text = ';'.join(['0.02', '1.0', str(1.0 / 90)])
        f.write(text)
        f.write(";")
        for item in l:
            f.write("%s" % item )
            if(l.index(item)+1!=size):
                f.write(",")
        f.write("\n")
# %%
l=list()
getIndex(train3, 100, 600, l)

# %%
for size in sizes:
    l = list()
    getIndex(train3, 100, size, l)
    train4 = train3.loc[l]
    train4=train4.drop('dist', axis=1)
    train4=train4.drop('ID', axis=1)
    train4=train4.drop('prediction', axis=1)


#%%


# Fit and predict on models of various training sizes
fit_list = list(map(lambda size: fit_svr(train3.iloc[:size]), sizes))
pred_list = list(map(lambda fit: fit.predict(valid2.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                     fit_list))
#%%
# Calculate R squared scores

r2 = list(map(lambda p: R2(p, valid2['winRate']), pred_list))
r2
#%%
_ = plt.plot(sizes, r2)
#%%
np.mean(r2)