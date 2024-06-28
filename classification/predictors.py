import xgboost as xgb
import numpy as np
import pandas as pd
import scipy
from typing import Callable, Any

from utils import *


class Predictor:

    def fit(self, Dtr: DTuple, Dva: DTuple):
        raise NotImplementedError()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """ 
        :return: np array of shape [n_samples, n_classes]
        """
        raise NotImplementedError()


class TreeEnsemble(Predictor):

    """ implements bootstrap ensemble """

    def __init__(self, tree_params, ens_size, subsampling_ratio):
        self.tree_params = tree_params
        self.ens_size = ens_size
        self.subsampling_ratio = subsampling_ratio
        self.ensemble = []

    def _fit_single(self, Dtr, Dva, params):
        early_stop = xgb.callback.EarlyStopping(
            rounds=10, metric_name='mlogloss', save_best=True)
        model = xgb.XGBClassifier(**params, enable_categorical=True, callbacks=[early_stop])
        model.fit(Dtr[0], Dtr[1], eval_set=[Dva], verbose=False)
        return model

    def fit(self, Dtr, Dva):
        seed = self.tree_params['seed']
        rng = np.random.default_rng(seed)
        for _ in range(self.ens_size):
            params = self.tree_params | {'seed': rng.integers(2**30)}
            Dtr_i = subsample_dtuple(Dtr, self.subsampling_ratio, rng)
            model = self._fit_single(Dtr_i, Dva, params)
            self.ensemble.append(model)

    def predict_proba(self, X, return_all_probs=False):
        all_pred_probs = []
        for  model in self.ensemble:
            all_pred_probs.append(model.predict_proba(X))
        all_pred_probs = np.asarray(all_pred_probs) # [n_models, n_samples, n_classes]
        if return_all_probs:
            return all_pred_probs
        return all_pred_probs.mean(axis=0)
    
    def get_fscores(self):
        ret = []
        for model in self.ensemble:
            ret.append(model.feature_importances_)
        return ret


def ipb_sample(
        Dtr: DTuple,
        Dva: DTuple,
        fit_fn: Callable[[DTuple, DTuple], Any],
        N_ratio: float,
        N_each_ratio: float,
        rng: np.random.Generator) -> Any:
    """ 
    sample a single IPB predictor 
    :param fit_fn: a function that takes Dtr, Dva and returns a sklearn-style classifier object
    """

    model = fit_fn(Dtr, Dva)
    n = Dtr[0].shape[0]
    N_tot, N_each = int(n*N_ratio), int(n*N_each_ratio)
    Nva_each = int(Dva[0].shape[0] * N_each_ratio)

    def extend(Dtup, delt_n):
        Xc, Yc = Dtup
        new_indices = polya_urn_sample(Xc.shape[0], delt_n, rng)
        if isinstance(Xc, np.ndarray):
            Xnew = Xc[new_indices]
        else:
            Xnew = Xc.iloc[new_indices]
        ynew_prob = model.predict_proba(Xnew)  # [delt_n, n_classes]
        Ynew = sample_categorical(ynew_prob, rng)
        Yret = np.concatenate([Yc, Ynew], 0)
        if isinstance(Xc, np.ndarray):
            return np.concatenate([Xc, Xnew], 0), Yret
        else:
            return pd.concat([Xc, Xnew], ignore_index=True), Yret

    for _ in range(0, N_tot, N_each):
        Dtr = extend(Dtr, N_each)
        Dva = extend(Dva, Nva_each)
        model = fit_fn(Dtr, Dva)

    return model
   

class TreeIPB(TreeEnsemble):
    
    """ IPB ensemble """ 

    def __init__(self, tree_params, ens_size, N_ratio, N_each_ratio):
        self.tree_params = tree_params
        self.ens_size = ens_size
        self.N_ratio = N_ratio
        self.N_each_ratio = N_each_ratio
        self.ensemble = []

    def _fit_single(self, Dtr, Dva, params, rng):
        cur_params = params | {'seed': rng.integers(2**30)}
        fit_fn = lambda dtr, dva: super(TreeIPB, self)._fit_single(dtr, dva, cur_params)
        return ipb_sample(Dtr, Dva, fit_fn, self.N_ratio, self.N_each_ratio, rng)

    def fit(self, Dtr, Dva):
        seed = self.tree_params['seed']
        rng = np.random.default_rng(seed)
        for i in range(self.ens_size):
            self.ensemble.append(self._fit_single(Dtr, Dva, self.tree_params, rng))
