# %%
# Load necessary packages

import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import matplotlib.pyplot as plt

# %%
# Read data and present

train = pd.read_csv('trainingData.csv')
valid = pd.read_csv('validationData.csv')
# %%
# Helper functions to preprocess data to bag-of-cards format

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
train = to_bag_of_cards(train)
valid = to_bag_of_cards(valid)
train.head()
clusters=(np.arange(2)+1)+217

#%%
for n in neigh:
    for c in clusters:
        print("Neighbouirs %s", n )
        print("Clusters %s", c )

        toy = train.copy(deep=False)
        max_size = 12000 + 1
        toy = toy[1:max_size]
        toy['cluster'] = 0
        toy['density'] = 0
        toy['distance'] = 0
        spectral, toy = cluster(toy, c)
        toy = calculate_distance(toy)
        solution = select_centers(toy,n)
        svr = fit_svm(toy.loc[solution])
        print(R2(svr.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)), valid['winRate']))
 # %%
# Sort data by number of games played

train = train.sort_values('nofGames', ascending=False)
valid = valid.sort_values('nofGames', ascending=False)

toy = train.copy(deep=False)

max_size = 12000 + 1
toy = toy[1:max_size]

toy['cluster'] = 0
toy['density'] = 0
toy['distance'] = 0

toy.shape
# solution = toy.iloc[:100].index.values.astype(int)
# %%
## Specify example model fitting function and R squared metric

from sklearn.svm import SVR


def R2(x, y):
    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y)))


def fit_svm(data):
    svr = SVR(kernel='rbf', gamma=1.0 / 90, C=1.0, epsilon=0.02, shrinking=False)
    svr.fit(data.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate', 'cluster', 'density', 'distance'], axis=1),
            data['winRate'])
    return svr

sizes = (np.arange(10) + 6) * 100
# %%
from sklearn.cluster import SpectralClustering
from sklearn import metrics


def cluster(data, n_of_clusters):
    X = data.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate', 'cluster', 'density', 'distance'], axis=1)

    db = SpectralClustering(affinity='rbf',
                            gamma=1.0 / 90,
                            n_clusters=n_of_clusters,
                            assign_labels="discretize",
                            random_state=0).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    data1 = data
    data1['cluster'] = db.labels_
    for cluster in np.unique(db.labels_):
        nOf = data1.loc[data['cluster'] == cluster].shape[0]
        data1.loc[data1['cluster'] == cluster, ['density']] = nOf

    return db, data1


# %%
spectral, toy = cluster(toy, 220)
# %%
toy.head()


# %%
def calculate_distance(df):
    # We select most dense clusters

    clusters = np.unique(df['cluster'])
    for cluster in clusters:
        df_cluster = df.loc[df['cluster'] == cluster].iloc[:, 4:94]
        df_dist_reduced = sp.spatial.distance.pdist(df_cluster, 'hamming')
        distance = sp.spatial.distance.squareform(df_dist_reduced).sum(axis=1)
        df.loc[df['cluster'] == cluster, 'distance'] = distance
        df.loc[df['cluster'] == cluster, 'density'] = df.loc[df[
                                                                 'cluster'] == cluster, 'density'] / sp.spatial.distance.squareform(
            df_dist_reduced).max()
    return df


# %%
toy = calculate_distance(toy)
# %%
toy.head()


# %%
def select_centers(df,c):
    clusters = np.unique(df['cluster'])
    solution = [100]
    for cluster in clusters:
        if df.loc[df['cluster'] == cluster].shape[0] > 60:
            idx = df[df.cluster == cluster].nsmallest(c, 'distance').index.values.astype(int)
            solution = np.append(solution, idx)
    print(solution.shape)
    print(type(solution))
    return solution


# %%
solution = select_centers(toy,5)
# %%
svr = fit_svm(toy.loc[solution])
print(R2(svr.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)), valid['winRate']))


# %%
def MCAL(df_clustered1, train_idx1):
    df_clustered = df_clustered1
    train_idx = train_idx1
    # the df_clustered contains columns cluster and density, and isSV
    # step 1


    # We train a svr on train_idx indices

    train_svr = df_clustered.loc[list(train_idx)]
    svr = fit_svm(train_svr)


    # Now we look for support vectors

    SV = svr.support_
    notSV = list(set(range(0, len(train_idx))) - set(SV))


    # Find clusters

    clusters = np.unique(df_clustered['cluster'])


    # We remove unimportant clusters

    for nonsv in notSV:

        idx = train_idx[nonsv]  # Now we retain indices of observations that are not support vectors

        key = df_clustered.index.get_loc(idx)  # And get its key
        unimportant = df_clustered.iloc[key]['cluster']  # label the cluster unimportant
        # df_clustered.iloc[key]['cluster'] = -1
        if np.isin(clusters, unimportant).any() == True:
            clusters = np.delete(clusters, np.where(clusters == unimportant))


    # Recalculate densities for important clusters (not important step)

    # for cluster in clusters:

    #    nOf = df_clustered.loc[df_clustered['cluster'] == cluster].shape[0]
    #    df_clustered.loc[df_clustered['cluster'] == cluster, ['density']] = nOf

    # Select new observations from important clusters

    df_pool = df_clustered.drop(train_idx, axis=0)  # We drop observations we already have
    df_pool = df_pool[df_pool.cluster.isin(clusters)]

    idx_clusters_number = min(max(int(len(clusters) * 0.9), min(10, len(clusters))), 20)

    idx_clusters = df_pool[['cluster', 'density']]
    idx_clusters = idx_clusters.drop_duplicates(subset='cluster')
    idx_clusters = idx_clusters.nlargest(columns='density', n=idx_clusters_number)
    idx_clusters = idx_clusters.cluster.unique()

    for clust in idx_clusters:

        dense = df_pool.loc[df_pool['cluster'] == clust, ['density']]  # Calculate density

        df_one = df_pool.loc[df_pool['cluster'] == clust].iloc[:, 4:94]  # Calculate distance
        df_dist_reduced = sp.spatial.distance.pdist(df_one, 'hamming')
        distance = sp.spatial.distance.squareform(df_dist_reduced).sum(axis=1)
        df_one.loc[df_pool['cluster'] == clust, 'distance'] = distance

        # Now what we want to choose ( tylko to zostalo)

        number_idx = df_pool[df_pool.cluster == clust].shape[0]
        if number_idx >= 2:
            number_idx = min(int(number_idx), 8)

        if number_idx >= 1:
            new_observation = df_one.nsmallest(number_idx, 'distance').index.values.astype(int)
            a = len(train_idx)
            train_idx = np.append(train_idx, new_observation)


    # for i in range(0,max_dense):
    #   new_observation_key = df_pool.loc[df_pool['cluster'] == cluster].index.values.astype(int)[0]
    #  train_idx = np.append(train_idx, df_pool.loc[df_pool['cluster'] == cluster].index.values.astype(int)[0])
    # df_pool = df_clustered.drop(df_pool.loc[df_pool['cluster'] == cluster].index.values.astype(int)[0],axis=0)

    return df_clustered, train_idx


# %%
def iterate_MCAL(df, solution, iterate):
    _, b = MCAL(toy, solution)
    for i in range(0, iterate):

        _, c = MCAL(toy, b)
        b = c
    return b


# %%
final = iterate_MCAL(toy, solution, 20)
# %%
train1 = toy.loc[final]
train1.head()
# %%
# Fit and predict on models of various training sizes

fit_list = list(map(lambda size: fit_svm(train1.iloc[:size]), sizes))
pred_list = list(map(lambda fit: fit.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                     fit_list))
# %%
# Calculate R squared scores

r2 = list(map(lambda p: R2(p, valid['winRate']), pred_list))
r2
# %%
_ = plt.plot(sizes, r2)
# %%
np.mean(r2)
# %%
# Save hyperparameteres and selected indices in submission format
train1 = toy.loc[final]
open('sub.txt', 'w').close()
with open('sub.txt', 'a') as f:
    for size in sizes:
        ind_text = ','.join(list(map(str, train1.index.values[:size])))
        text = ';'.join(['0.02', '1.0', str(1.0 / 90), ind_text])
        f.write(text + '\n')
# %%
spectral, toy = cluster(toy, 219)
toy = calculate_distance(toy)
solution = select_centers(toy,9)
svr = fit_svm(toy.loc[solution])
print(R2(svr.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)), valid['winRate']))
#%%
final = iterate_MCAL(toy, solution, 15)
train1 = toy.loc[final]

fit_list = list(map(lambda size: fit_svm(train1.iloc[:size]), sizes))
pred_list = list(map(lambda fit: fit.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                     fit_list))

r2 = list(map(lambda p: R2(p, valid['winRate']), pred_list))
#%%
final=final.tolist()
l=train.index
for i in train:
    if i not in final:
        final.append(i)

#%%
clusters=(np.arange(1)+1)+215
neigh=np.arange(6)+5
train = train.sort_values('nofGames', ascending=False)

for c in clusters:
    toy = train.copy(deep=False)
    max_size = 10000 + 1
    toy = toy[1:max_size]
    toy['cluster'] = 0
    toy['density'] = 0
    toy['distance'] = 0
    spectral, toy = cluster(toy, c)
    toy = calculate_distance(toy)
    for size in sizes:
        for n in neigh:
            print(size, n)
            solution = select_centers(toy, n)
            svr = fit_svm(toy.loc[solution])
            final = iterate_MCAL(toy, solution, 20)
            train1 = toy.loc[final]

            fit_list = fit_svm(train1.iloc[:size])

            pred_list = fit_list.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1))

            r2 = R2(pred_list, valid['winRate'])
            df2 = pd.DataFrame([[c, size, n, r2]], columns=['clusters', 'size', 'neigh','R2'])
            results=results.append(df2, ignore_index=True)




#%%



for n in neigh:
    print(n)
    solution = select_centers(toy,n)

    svr = fit_svm(toy.loc[solution])
    print(R2(svr.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)), valid['winRate']))
    final = iterate_MCAL(toy, solution, 20)
    train1 = toy.loc[final]

    fit_list = list(map(lambda size: fit_svm(train1.iloc[:size]), sizes))
    pred_list = list(map(lambda fit: fit.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                     fit_list))

    r2 = R2(pred_list, valid['winRate'])
    print(r2)
#%%
best = pd.DataFrame(columns=['clusters', 'size', 'neigh', 'R2'])
for size in sizes:
    r=results[results['size']==size]
    r = r.sort_values('R2', ascending=False)
    best = best.append(r.iloc[0,:], ignore_index=True)
    best['clusters'] = best['clusters'].astype('int')
    best['size'] = best['size'].astype('int')
    best['neigh'] = best['neigh'].astype('int')
c=best['clusters'].unique()
#%%
size3=list()
samples=list()
for n_of_clusters in c:
    toy = train.copy(deep=False)
    max_size = 10000 + 1
    toy = toy[1:max_size]
    toy['cluster'] = 0
    toy['density'] = 0
    toy['distance'] = 0
    spectral, toy = cluster(toy, n_of_clusters)
    toy = calculate_distance(toy)
    bestC = best[best['clusters'] == n_of_clusters]
    size2 = bestC['size']
    for size in size2:
        bestS = best[best['size'] == size]
        n = bestS.iloc[0, 2]
        print(size, n)
        solution = select_centers(toy, n)
        svr = fit_svm(toy.loc[solution])
        final = iterate_MCAL(toy, solution, 20)
        final=np.array(final)
        final=final[:size]
        train1 = toy.loc[final]
        samples.append(final)
        size3.append(size)

#%%
np.savetxt('param3.txt', results.values, fmt='%f')

#%%
results=pd.read_csv('param3.txt', header=None, sep=" ")
results.columns=['clusters', 'size', 'neigh', 'R2']
results['clusters']=results['clusters'].astype('int64')
results['size']=results['size'].astype('int64')
results['neigh']=results['neigh'].astype('int64')
#%%
for i in samples:
    i=i.tolist()

#%%
samples=sorted(samples, key=len)
#%%
#%%
train1 = toy.loc[final]
open('sub.txt', 'w').close()
with open('sub.txt', 'a') as f:
    for size in sizes:
        text = ';'.join(['0.02', '1.0', str(1.0 / 90)])
        f.write(text)
        f.write(';')
        l=samples[int(size / 100 - 6)]
        l=l.tolist()
        for item in l:
            f.write("%s" % item)
            if (l.index(item) + 1 != size):
                f.write(",")
        f.write("\n")
#%%
plik=open('sub.txt')
listalist=list()
for i in plik:
    i=i[i.rfind(';')+1:-1:]
    i=i.split(',')
    listalist.append(i)
plik.close()
for i in listalist:
    results = list(map(int, i))
#%%
listalist2=list()
for i in listalist:
    listalist2.append(list(map(int, i)))
listalist=listalist2
#%%

def fit_svm2(data,gamma,C,e):
    svr = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=e, shrinking=False)
    svr.fit(data.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate', 'cluster', 'density', 'distance'], axis=1),
            data['winRate'])
    return svr


gamma=1.0/(np.arange(1)+45)
epsilon=(np.arange(2))/500+0.02
C=(np.arange(2))/10+1.2

final=listalist[9]


#%%

train = train.sort_values('nofGames', ascending=False)
valid = valid.sort_values('nofGames', ascending=False)
toy = train.copy(deep=False)
max_size = 12000 + 1
toy['cluster'] = 0
toy['density'] = 0
toy['distance'] = 0
train1 = toy.loc[final]


#%%
for g in gamma:
    for cp in C:
        for e in epsilon:
            for size in sizes:
                train1 = toy.loc[final]
                fit = fit_svm2(train1.iloc[:size],g,cp,e)
                valid2 = valid.iloc[:, 4:]
                pred= fit.predict(valid2)
                r2 = R2(pred, valid['winRate'])
                df = pd.DataFrame([[g,cp,e,size, np.mean(r2)]], columns=['gamma','C','eps', 'size','R2'])
                results = results.append(df, ignore_index=True)


#%%
l2=list()
#%%
for size in sizes:
    df=results[results['size']==size]
    df = df.sort_values('R2', ascending=False)
    r=df.iloc[0,4]
    l.append(r)

#%%
results = pd.DataFrame( columns=['gamma','C','eps', 'size','R2'])
#%%
np.savetxt('param5 .txt', results.values, fmt='%f')
#%%
results2=results
#%%
print(sum(l)/len(l))
print(sum(l2)/len(l2))

#%%
for size in sizes:
    train1 = toy.loc[final]
    fit = fit_svm(train1.iloc[:size])
    valid2 = valid.iloc[:, 4:]
    pred = fit.predict(valid2)
    r= R2(pred, valid['winRate'])
    l2.append(r)
#%%
for size in sizes:
    df=results[results['size']==size]
    df = df.sort_values('R2', ascending=False)
    print(size)
    print('eps', df.iloc[0,2])
    print('C', df.iloc[0,1])
    print('gamma', df.iloc[0,0])


