import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _to_onehot(x):
    n = x.max() + 1
    return np.eye(n)[x]


def proc_pd(xall):
    ret = {}
    for c in xall.columns:
        if xall[c].dtype == np.object_:
            ret[c] = xall[c].astype('category')
        else:
            cc = xall[c].astype('float')
            ret[c] = (cc - cc.mean()) / (cc.std() + 1e-8)
    return pd.DataFrame(ret)


def load_adult(return_pd=True):
    df = pd.read_csv('data/adult.data', header = None)
    df = pd.concat([df, pd.read_csv('data/adult.test', header = None)])
    if return_pd:
        xall = proc_pd(df.iloc[:, :14])
    else:
        Dall = []
        df[13]=np.where(df[13]=='United-States',1,0)
        for c in df.columns[:-1]:
            if c == 3:  # drop education as we have education-num
                continue
            if df[c].dtype == np.object_:
                x = LabelEncoder().fit_transform(df[c])
                # Dall.append(x[:, None]) 
                Dall.append(_to_onehot(x))
            else:
                Dall.append(df[c].to_numpy()[:, None])
        Dall = np.concatenate(Dall, axis=1).astype('f')
        xall = Dall
        xall = (xall - xall.mean(axis=0)) / (xall.std(axis=0)+1e-8)
    yall = (df[14].to_numpy() == ' >50K').astype('i')
    return xall, yall

# 4 UCI datasets from QianLab/NR_SMOCU_SGD_GPC

def load_wine(return_pd=True):
    df = pd.read_csv('data/wine.data.csv', header = None)
    temparray = [1, 2]
    df2 = df[df[0].isin(temparray)]
    #df2 = df[~df[6].isin(['?'])]
    if return_pd:
        xall = proc_pd(df2.iloc[:, 1:14])
    else:
        xall2 = df2.iloc[:, 1:14].to_numpy().astype(float)
        xall = (xall2 - xall2.mean(axis = 0))/(xall2.std(axis = 0)+1e-8)
    yall = (df2.iloc[:, 0].to_numpy() == temparray[0]).astype(int)
    return xall, yall


def load_vehicle(return_pd=True):
    filelist = "abcdefghi"
    pieces = []
    for file in filelist:
        df = pd.read_csv('data/vehicle/xa'+file+'.dat', delim_whitespace = True
                     ,header = None)
        pieces.append(df)
        
    df = pd.concat(pieces)
    df2 = df[df[18].isin(['van', 'saab'])]
    #df2 = df[~df[6].isin(['?'])]
    if return_pd:
        xall = proc_pd(df2.iloc[:, 0:18])
    else:
        xall2 = df2.iloc[:, 0:18].to_numpy().astype(float)
        xall = (xall2 - xall2.mean(axis = 0))/(xall2.std(axis = 0)+1e-8)
    yall = (df2.iloc[:, 18].to_numpy() == 'van').astype(int)
    return xall, yall


def load_wdbc(return_pd=True):
    df = pd.read_csv('data/wdbc.data', header = None)
    #df2 = df[~df[6].isin(['?'])]
    if return_pd:
        xall = proc_pd(df.iloc[:, 2:])
    else:
        xall2 = df.iloc[:, 2:].to_numpy().astype(float)
        xall = (xall2 - xall2.mean(axis = 0))/(xall2.std(axis = 0)+1e-8)
    yall = (df.iloc[:, 1].to_numpy() == 'M').astype(int)
    return xall, yall


def load_ionosphere(return_pd=True):
    df = pd.read_csv('data/ionosphere.data.csv', header=None)
    df2 = df
    if return_pd:
        xall = proc_pd(df2.iloc[:, 0:34])
    else:
        xall2 = df2.iloc[:, 0:34].to_numpy().astype(float)
        xall = (xall2 - xall2.mean(axis=0))/(xall2.std(axis=0) + 1e-8)
        xall[:, 0:2] = xall2[:, 0:2]
    yall = (df.iloc[:, 34].to_numpy() == 'g').astype(int)
    return xall, yall


def load_cc18(data_rank: int, base_path='data/cc18_subset'):
    """
    load the smaller subset of OpenML-CC18 following the TabPFN paper.
    NOTE: the CSVs in `base_path` are named using the task IDs, which are different from the 
          data IDs in the TabPFN repo
    """
    paths = [os.path.join(base_path, p) for p in sorted(os.listdir(base_path))]
    assert len(paths) == 30
    df = pd.read_csv(paths[data_rank])
    xall = proc_pd(df.iloc[:, :-1])
    assert df.iloc[:, -1].name == 'TARGET'
    yall_onehot = df.iloc[:, -1].astype('str').str.get_dummies()  # [n_samples, n_classes]
    yall = np.argmax(yall_onehot.values, axis=1)
    return xall, yall


def load(dname, seed):
    if dname.startswith('cc18'):
        dId = int(dname.split('_')[1])
        xall, yall = load_cc18(dId)
    else:
        xall, yall = globals()['load_' + dname]()
    xspace, Xtest, yspace, ytest = train_test_split(xall, yall, test_size=0.2, random_state=seed)
    Xtr, Xval, ytr, yval = train_test_split(xspace, yspace, test_size=0.25, random_state=seed*2+1)
    return (Xtr, ytr), (Xval, yval), (Xtest, ytest)
