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
train = to_bag_of_cards(train)
valid = to_bag_of_cards(valid)
train.head()
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

def cluster(data, n_of_clusters, idx_SV):
    X = data.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate', 'cluster', 'density', 'distance'], axis=1)
    X = X.drop(idx_SV, axis=0)

    db = SpectralClustering(affinity='rbf',
                            gamma=1.0 / 90,
                            n_clusters=n_of_clusters,
                            assign_labels="discretize",
                            random_state=0).fit(X)

    # db = KMeans(n_clusters=100, random_state=0).fit(X)

    # df.loc[df['cluster'] == cluster, 'density'] = df.loc[df['cluster'] == cluster, 'density']
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    # data1 = data
    data1 = data
    data1['density'] = 0

    data1.loc[~data1.index.isin(idx_SV), 'cluster'] = db.labels_  # nadac klastry tylko tym ktore nie sa w idx_SV
    data1.loc[data1.index.isin(idx_SV), 'cluster'] = -1
    # data1.loc[idx_SV]['cluster'] = -1

    for cluster in np.unique(db.labels_):
        nOf = data1.loc[data1['cluster'] == cluster].shape[0]
        data1.loc[data1['cluster'] == cluster, ['density']] = nOf

    return db, data1

def calculate_distance(df):
    # We select most dense clusters
    df['distance'] = 0
    clusters = np.unique(df['cluster'])

    for cluster in clusters:
        # df_cluster = df.loc[df['cluster'] == cluster].iloc[:,4:94]
        df_cluster = df.loc[df['cluster'] == cluster].drop(
            ['deck', 'nofGames', 'nOfPlayers', 'winRate', 'cluster', 'density', 'distance'], axis=1)
        df_dist_reduced = sp.spatial.distance.pdist(df_cluster, 'hamming')
        distance = sp.spatial.distance.squareform(df_dist_reduced).sum(axis=1)
        df.loc[df['cluster'] == cluster, 'distance'] = distance

        max_distance = df.loc[df['cluster'] == cluster, 'distance'].max()
        frequency = df.loc[df['cluster'] == cluster].shape[0]

        if max_distance > 0:
            df.loc[df['cluster'] == cluster, 'density'] = frequency / max_distance

    return df
def select_centers(df, size, sample):
    # clusters = np.unique(df['cluster'])
    a = df.density.describe()[4]
    selector = df[['cluster', 'density']]
    selector = selector.drop_duplicates('cluster')
    selector = selector[selector['density'] > a]
    # selector = selector.nsmallest(100,'distance')
    selector = np.array(selector['cluster'])
    clusters = selector

    solution = [100]

    for cluster in clusters:
        if df.loc[df['cluster'] == cluster].shape[0] > size:
            idx = df[df.cluster == cluster].nsmallest(sample, 'distance').index.values.astype(int)
            solution = np.append(solution, idx)

    print(len(solution))
    return solution[1:]


# %%
def baseline(df_PCA, df, solution, n_of_clusters, size, sample):
    train_svr = df.loc[solution]
    svr = fit_svm(train_svr)
    print("Calculated R2 on validation set is:",
          np.round(R2(svr.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)), valid['winRate']),
                   5))

    SV = svr.support_
    idx_SV = solution[SV]  # Te indeksy wyrzucamy z df_PCA
    # df_PCA = df_PCA.drop(idx_SV, axis=0)

    _, df_PCA = cluster(df_PCA, n_of_clusters, idx_SV)
    df_PCA = calculate_distance(df_PCA)

    notSV = list(set(range(0, len(solution))) - set(SV))
    idx_notSV = solution[notSV]
    print(f"Number of observations that are not support vectors is {len(notSV)} out of {len(notSV) + len(SV)}")

    clusters_list = np.unique(df_PCA['cluster'])
    clusters_list = np.delete(clusters_list, np.where(clusters_list == -1))

    for idx in idx_notSV:

        row_number = df_PCA.index.get_loc(idx)
        unimportant = df_PCA.iloc[row_number]['cluster']

        if np.isin(clusters_list, unimportant).any() == True:
            clusters_list = np.delete(clusters_list, np.where(clusters_list == unimportant))

    df_select_centers = df_PCA[df_PCA['cluster'].isin(clusters_list)]
    solution_new = select_centers(df_select_centers, size, sample)
    solution = np.append(solution, solution_new)
    print(len(solution))

    return df_PCA, solution


# %%
final = solution
# %%
svr.support_final = final[1:]
# %%
train1 = X.loc[final]
train1.head()
# %%
# Fit and predict on models of various training sizes

train1 = X.loc[final]
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

with open('sub_python4.txt', 'a') as f:
    for size in sizes:
        ind_text = ','.join(list(map(str, train1.index.values[:size])))
        text = ';'.join(['0.02', '1.0', str(1.0 / 90), ind_text])
        f.write(text + '\n')
#%%
results = pd.DataFrame(columns=['size_train', 'clusters', 'big','sample','len', 'R2','first'])
# %%
size_train=(np.arange(2)+1)*5000
clusters=(np.arange(4)+1)*50
big=(np.arange(5)+1)*10
sample=(np.arange(3)+1)*5
#%%
train = train.sort_values('nofGames', ascending=False)
train_large = train.iloc[10000:15000]
#%%
for s in size_train:
    for c in clusters:
        for b in big:
            for s in sample:
                X = train.iloc[0:s]
                X['cluster'] = 0
                X['density'] = 0
                X['distance'] = 0
                solution = X.iloc[0:100].index.astype(int)
                X_new, solution = baseline(X, X, solution, c, b, s)
                for i in range(7):
                    X_new, solution = baseline(X_new, X, solution, c, b, s)
                    X_additional = train_large.sample(frac=len(solution) / X.shape[0])
                    X = X.append(X_additional, ignore_index=False)
                train1 = X.loc[solution]
                fit_list = list(map(lambda size: fit_svm(train1.iloc[:size]), sizes))
                pred_list = list(
                    map(lambda fit: fit.predict(valid.drop(['deck', 'nofGames', 'nOfPlayers', 'winRate'], axis=1)),
                        fit_list))
                r2 = list(map(lambda p: R2(p, valid['winRate']), pred_list))

                df2 = pd.DataFrame([[s, c, b, s,len(sample),np.mean(r2),100]], columns=['size_train', 'clusters', 'big','sample','len', 'R2','first'])
                results.append(df2, ignore_index=True)


# %%
_, X = cluster(X, 200, [])
# %%
solution = select_centers(X, 70, 10)
# %%
#%%
train = train.sort_values('nofGames', ascending=False)

train_large = train.iloc[10000:30000]

X = train.copy(deep=False)
X = X.iloc[0:10000]
solution = X.iloc[0:30].index.astype(int)

X['cluster'] = 0
X['density'] = 0
X['distance'] = 0

#%%
_,X = cluster(X,200,[])
#%%
solution = select_centers(X,70,10)
#%%
X_new, solution = baseline(X,X,solution,50,70,10)
#%%
for i in range(5):
    X_new, solution = baseline(X_new,X,solution,50,70,10)
    X_additional = train_large.sample(frac=len(solution)/X.shape[0])
    X = X.append(X_additional, ignore_index=False)