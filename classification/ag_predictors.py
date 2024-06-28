import os, tempfile
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import scipy
import pickle

from predictors import Predictor, ipb_sample
from utils import *


def dtuple_to_df(dtup):
    return pd.concat([dtup[0], pd.Series(dtup[1], index=dtup[0].index, name='TARGET')], axis=1) 


class AGBasePredictor(Predictor):

    def __init__(self, ag_preset, n_threads):
        self._presets = [ag_preset, 'optimize_for_deployment']
        self._n_threads = n_threads
        self._ag_path = tempfile.TemporaryDirectory()
        assert len(os.listdir(self._ag_path.name)) == 0, 'Temp dir not empty'
        self._predictor = TabularPredictor(
            label='TARGET', problem_type='multiclass', path=self._ag_path.name,
            verbosity=1, eval_metric='log_loss')

    def fit(self, Dtr, Dva):
        self._predictor = self._predictor.fit(
            train_data=dtuple_to_df(Dtr), tuning_data=dtuple_to_df(Dva), presets=self._presets,
            num_cpus=self._n_threads, use_bag_holdout=True, ds_args={'memory_safe_fits': False})

    def predict_proba(self, X):
        rdf = self._predictor.predict_proba(X)
        return rdf.values
    

class AGIPBPredictor(Predictor):

    def __init__(self, ag_preset, n_threads, ens_size, ipb_n_each, ipb_n_tot, seed):
        self._ag_preset, self._n_threads = ag_preset, n_threads
        self._ens_size, self._n_each, self._n_tot = ens_size, ipb_n_each, ipb_n_tot
        self._rng = np.random.default_rng(seed)
        self.ensemble = []

    def _fit_single(self, Dtr, Dva):

        def _fit_fn(d_tr, d_va):
            bp = AGBasePredictor(self._ag_preset, self._n_threads)
            bp.fit(d_tr, d_va)
            return bp
            
        if self._n_tot == 0:  # bagging 
            Dtr = subsample_dtuple(Dtr, -1, self._rng)
            return _fit_fn(Dtr, Dva)

        return ipb_sample(Dtr, Dva, _fit_fn, self._n_tot, self._n_each, self._rng)

    def fit(self, Dtr, Dva):
        for _ in range(self._ens_size):
            self.ensemble.append(self._fit_single(Dtr, Dva))

    def predict_proba(self, X, return_all_probs=False):
        all_pred_probs = []
        for  model in self.ensemble:
            all_pred_probs.append(model.predict_proba(X))
        all_pred_probs = np.asarray(all_pred_probs) # [n_models, n_samples, n_classes]
        if return_all_probs:
            return all_pred_probs
        return all_pred_probs.mean(axis=0)
 
