#%%
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.svm import SVR
import pandas as pd
import numpy as np


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


train2 = to_bag_of_cards(train)
valid2 = to_bag_of_cards(valid)
train2.head()


def R2(x, y):
    return 1 - np.sum(np.square(x - y)) / np.sum(np.square(y - np.mean(y))) - mean_squared_error(x, y)

def fit_svr(data):
    svr = SVR(kernel='rbf', gamma=1.0/90, C=2.0, epsilon=0.02, shrinking=False)
    svr.fit(data, data['winRate'])
    return svr

sizes = (np.arange(10) + 6) * 100

#%%
train2=train2.iloc[:,3:]
#%%
learner = ActiveLearner(
    estimator=SVR(),
    X_training=train2.iloc[:,1:], y_training=t
)
#%%
t=train2.iloc[:,:1].to_numpy()
