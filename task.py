#%%
# Load necessary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn import svm
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
    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y))) - mean_squared_error(x, y)

def fit_svr(data):
    svr = SVR(kernel='rbf', gamma=1.0/90, C=2.0, epsilon=0.02, shrinking=False)
    svr.fit(data, data['winRate'])
    return svr

sizes = (np.arange(10) + 6) * 100

#%%
train3=train2.iloc[:,3:]
#%%
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
trainingsets=list()
for size in sizes:
    l = list()
    getIndex(train3, 100, size, l)
    train4 = train3.loc[l]
    train4=train4.drop('dist', axis=1)
    train4=train4.drop('ID', axis=1)
    train4=train4.drop('prediction', axis=1)
    fit_list = list(map(lambda size: fit_svr(train4), sizes))

# %%
valid3=valid2.iloc[:,3:]
r2=list()
for size in sizes:
    l = list()
    getIndex(train3, 100, size, l)
    train4 = train3.loc[l]
    train4=train4.drop('dist', axis=1)
    train4=train4.drop('ID', axis=1)
    train4=train4.drop('prediction', axis=1)
    fit = fit_svr(train4)
    pred=fit.predict(valid3)
    r=R2(pred, valid2['winRate'])
    r2.append(tuple((size, r)))
plt.scatter(*zip(*r2))
plt.show()
print(mean([x[1] for x in r2]))


#%%
for nClusters in clusters:
    train3=train2.iloc[:, 3:]
    kmeans = KMeans(n_clusters=int(nClusters))
    kmeans.fit(train3)
    y_km = kmeans.fit_predict(train3)
    train3_dist=kmeans.transform(train3)
    train3['dist']=train3_dist.min(axis=1)
    train3['prediction'] = y_km
    train3['ID'] = train3.index
    trainingsets=list()
    for size in sizes:
        l = list()
        getIndex(train3, nClusters, size, l)
        train4 = train3.loc[l]
        train4=train4.drop('dist', axis=1)
        train4=train4.drop('ID', axis=1)
        train4=train4.drop('prediction', axis=1)
        fit_list = list(map(lambda size: fit_svr(train4), sizes))
    valid3=valid2.iloc[:,3:]
    r2=list()
    for size in sizes:
        l = list()
        getIndex(train3, nClusters, size, l)
        train4 = train3.loc[l]
        train4=train4.drop('dist', axis=1)
        train4=train4.drop('ID', axis=1)
        train4=train4.drop('prediction', axis=1)
        fit = fit_svr(train4)
        pred=fit.predict(valid3)
        r=R2(pred, valid2['winRate'])
        r2.append(tuple((size, r)))
    plt.scatter(*zip(*r2))
    plt.show()
    print(nClusters)
    print(mean([x[1] for x in r2]))

    open('clusters.txt', 'w').close()

    with open('clusters.txt', 'a') as f:
        for size in sizes:
            l = list()
            getIndex(train3, nClusters, size, l)
            text = ';'.join(['0.02', '1.0', str(1.0 / 90)])
            f.write(text)
            f.write(";")
            for item in l:
                f.write("%s" % item)
                if (l.index(item) + 1 != size):
                    f.write(",")
            f.write("\n")

#%%
s=[1400,1500]
r2=list()
indexes2=list()
for size in s:
    #ustawienie kmeans i obliczenie odległości od klastra
    train3 = train2.iloc[:, 3:]
    kmeans = KMeans(n_clusters=int(size))
    kmeans.fit(train3)
    y_km = kmeans.fit_predict(train3)
    train3_dist = kmeans.transform(train3)
    train3['dist'] = train3_dist.min(axis=1)
    train3['prediction'] = y_km
    train3['ID'] = train3.index
    #dostajemy po jednym elemencie z każdego klastra
    l = list()
    getIndex(train3, size, size, l)
    indexes2.append(l)
    train4 = train3.loc[l]
    train4=train4.drop('dist', axis=1)
    train4=train4.drop('ID', axis=1)
    train4=train4.drop('prediction', axis=1)
    fit=fit_svr(train4)
    valid3=valid2.iloc[:,3:]
    pred=fit.predict(valid3)
    r=R2(pred, valid2['winRate'])
    r2.append(tuple((size, r)))

#%%

open('clusters2.txt', 'w').close()
with open('clusters2.txt', 'a') as f:
    for i in range(8):
        l = indexes[i]
        text = ';'.join(['0.02', '1.0', str(1.0 / 90)])
        f.write(text)
        f.write(";")
        for item in l:
            f.write("%s" % item)
            if (l.index(item) + 1 != size):
                f.write(",")
        f.write("\n")

#%%
indexes3=indexes

s=[1400,1500]
for size in sizes:
    l = list()
    getIndex(train3, 100, size, l)
    indexes3.append(l)

#%%
file=open('clusters2.txt')
training=list()
for i in file:
    i=i[i.rfind(';')+1:-1:]
    i=i.split(',')
    training.append(i)
file.close()
#%%
def fit_svr2(data, gamma, epsilon):
    svr = SVR(kernel='rbf', gamma=gamma, C=1.0, epsilon=epsilon, shrinking=False)
    svr.fit(data, data['winRate'])
    return svr

#%%
index = training[int(600 / 100 - 6)]
index = map(int, index)

t = train3.loc[index]
results=pd.DataFrame(columns=['gamma', 'epsilon', 'R2','size'])

#%%
gamma = 1/(np.arange(10) + 2)
eps=(np.arange(10) + 1) / 100
for size in sizes:
    index = training[int(size / 100 - 6)]
    index = map(int, index)
    t = train3.loc[index]
    for epsilon in eps:
        for g in gamma:
            fit=fit_svr2(t,g,epsilon)
            valid3=valid2.iloc[:,3:]
            pred=fit.predict(valid3)
            r=R2(pred, valid2['winRate'])
            df2 = pd.DataFrame([[g,epsilon,r, size]], columns=['gamma', 'epsilon', 'R2', 'size'])
            results=results.append(df2, ignore_index=True)

#%%
gamma2=list()
epsilon2=list()
open('clusters4.txt', 'w').close()
with open('clusters4.txt', 'a') as f:
    for size in sizes:
        print(size)
        resultsSize=results[results['size'] == size]
        resultsSize=resultsSize.sort_values(by='R2', axis=0, ascending=False)
        r = resultsSize.iloc[0, :]
        print("gamma:")
        print(str(r['gamma']))
        print("epsilon:")
        print(str(r['epsilon']))
        print(str(r['R2']))

        f.write(str(r['epsilon']))
        f.write(";1.0;")
        f.write(str(r['gamma']))
        f.write(";")
        l=training[int(size / 100 - 6)]
        for item in l:
            f.write("%s" % item)
            if (l.index(item) + 1 != size):
                f.write(",")
        f.write("\n")



#%%
gamma2=list()
epsilon2=list()
open('clusters5.txt', 'w').close()
with open('clusters5.txt', 'a') as f:
    for size in sizes:
        print(size)
        resultsSize=results[results['size'] == size]
        resultsSize=resultsSize.sort_values(by='R2', axis=0, ascending=False)
        r = resultsSize.iloc[1, :]
        print("gamma:")
        print(str(r['gamma']))
        print("epsilon:")
        print(str(r['epsilon']))
        f.write(str(r['epsilon']))
        f.write(";1.0;")
        f.write(str(r['gamma']))
        f.write(";")
        l=training[int(size / 100 - 6)]
        for item in l:
            f.write("%s" % item)
            if (l.index(item) + 1 != size):
                f.write(",")
        f.write("\n")

#%%

gamma2=list()
epsilon2=list()
open('clusters6.txt', 'w').close()
with open('clusters6.txt', 'a') as f:
    for size in sizes:
        print(size)
        resultsSize=results[results['size'] == size]
        resultsSize=resultsSize.sort_values(by='R2', axis=0, ascending=False)
        r = resultsSize.iloc[2, :]
        print("gamma:")
        print(str(r['gamma']))
        print("epsilon:")
        print(str(r['epsilon']))
        f.write(str(r['epsilon']))
        f.write(";1.0;")
        f.write(str(r['gamma']))
        f.write(";")
        l=training[int(size / 100 - 6)]
        for item in l:
            f.write("%s" % item)
            if (l.index(item) + 1 != size):
                f.write(",")
        f.write("\n")

