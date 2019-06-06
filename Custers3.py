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
train.head()
# %%
valid.head()


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

# %%
# Sort data by number of games played

train = train.sort_values('nofGames', ascending=False)
valid = valid.sort_values('nofGames', ascending=False)

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

def select_centers(df, n):
    clusters = np.unique(df['cluster'])
    solution = [100]
    for cluster in clusters:
        if df.loc[df['cluster'] == cluster].shape[0] > 60:
            idx = df[df.cluster == cluster].nsmallest(n, 'distance').index.values.astype(int)
            solution = np.append(solution, idx)
    print(solution.shape)
    print(type(solution))
    return solution

# %%
neigh=np.arange(3)+3

clusters=(np.arange(8)+1)*50

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
def MCAL(df_clustered1, train_idx1):
    df_clustered = df_clustered1
    train_idx = train_idx1
    # the df_clustered contains columns cluster and density, and isSV
    # step 1
    print("-------------------------------------------------------------")
    print(f"Number of samples: {len(train_idx)}")

    # We train a svr on train_idx indices

    train_svr = df_clustered.loc[list(train_idx)]
    svr = fit_svm(train_svr)
    print("Calculated R2 on validation set is:",
          np.round(R2(svr.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)), valid['winRate']),
                   5))

    # Now we look for support vectors

    SV = svr.support_
    notSV = list(set(range(0, len(train_idx))) - set(SV))

    print(f"Number of observations that are not support vectors is {len(notSV)} out of {len(notSV) + len(SV)}")

    # Find clusters

    clusters = np.unique(df_clustered['cluster'])

    print(f"There are {len(clusters)} initial clusters")

    # We remove unimportant clusters

    for nonsv in notSV:

        idx = train_idx[nonsv]  # Now we retain indices of observations that are not support vectors

        key = df_clustered.index.get_loc(idx)  # And get its key
        unimportant = df_clustered.iloc[key]['cluster']  # label the cluster unimportant
        # df_clustered.iloc[key]['cluster'] = -1
        if np.isin(clusters, unimportant).any() == True:
            clusters = np.delete(clusters, np.where(clusters == unimportant))

    print(f"There are {len(clusters)} important clusters")

    # Recalculate densities for important clusters (not important step)

    # for cluster in clusters:

    #    nOf = df_clustered.loc[df_clustered['cluster'] == cluster].shape[0]
    #    df_clustered.loc[df_clustered['cluster'] == cluster, ['density']] = nOf

    # Select new observations from important clusters

    df_pool = df_clustered.drop(train_idx, axis=0)  # We drop observations we already have
    df_pool = df_pool[df_pool.cluster.isin(clusters)]

    new_idx_number = 100  # We want that much new training samples
    idx_clusters_number = min(max(int(len(clusters) * 0.8), min(5, len(clusters))), 20)

    idx_clusters = df_pool[['cluster', 'density']]
    idx_clusters = idx_clusters.drop_duplicates(subset='cluster')
    idx_clusters = idx_clusters.nlargest(columns='density', n=idx_clusters_number)
    idx_clusters = idx_clusters.cluster.unique()
    print(f"We choose {len(idx_clusters)} clusters.")

    for clust in idx_clusters:

        dense = df_pool.loc[df_pool['cluster'] == clust, ['density']]  # Calculate density
        max_dense = int(dense.iloc[0])

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
            print(new_observation)
            a = len(train_idx)
            train_idx = np.append(train_idx, new_observation)
    print(f"We have {len(train_idx)} observations after this iteration.")
    print("------------------------------------------------------------")

    # for i in range(0,max_dense):
    #   new_observation_key = df_pool.loc[df_pool['cluster'] == cluster].index.values.astype(int)[0]
    #  train_idx = np.append(train_idx, df_pool.loc[df_pool['cluster'] == cluster].index.values.astype(int)[0])
    # df_pool = df_clustered.drop(df_pool.loc[df_pool['cluster'] == cluster].index.values.astype(int)[0],axis=0)

    return df_clustered, train_idx


# %%
def iterate_MCAL(df, solution, iterate):
    _, b = MCAL(toy, solution)
    for i in range(0, iterate):
        print("-------------------------------------------------------")
        print(f"---------------------Iteration no {i} -----------------")
        print("-------------------------------------------------------")
        _, c = MCAL(toy, b)
        b = c
    return b


# %%
final = iterate_MCAL(toy, solution, 20)
# %%
final[0:10]
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

with open('sub_python1.txt', 'a') as f:
    for size in sizes:
        ind_text = ','.join(list(map(str, train1.index.values[:size])))
        text = ';'.join(['0.02', '1.0', str(1.0 / 90), ind_text])
        f.write(text + '\n')
# %%
