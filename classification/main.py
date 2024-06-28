import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy
import os, pickle
from typing import Tuple

import exputils, datasets
from predictors import *
from utils import *


def get_parser():
    parser = exputils.parser('tree')
    parser.add_argument('--dataset', default='ionosphere', type=str)
    parser.add_argument('--data_seed', default=0, type=int)
    parser.add_argument('--train_seed', default=0, type=int)
    #
    parser.add_argument('--method', type=str, default='tree')
    parser.add_argument('--n_threads', default=4, type=int)
    # quality preset in AutoGluon (see Appendix D.4)
    parser.add_argument('--ag_preset', type=str, default='medium_quality')
    # tree hyperparameters
    parser.add_argument('--subsample_n', default=-1, type=int)
    parser.add_argument('--max_depth', default=3, type=int)
    parser.add_argument('--n_estimators', default=100, type=int)
    parser.add_argument('--lr', default=0.3, type=float)
    parser.add_argument('--bagging_ratio', type=float, default=-1)
    # IPB hyperparameters
    parser.add_argument('--ipb_N_ratio', type=float, default=1)
    parser.add_argument('--ipb_N_each', type=float, default=1)
    parser.add_argument('--ens_size', type=int, default=20)
    
    return parser


def calc_ece(pred_probs, y, n_bins=8):
    """ ECE for binary classification """
    assert pred_probs.ndim == y.ndim == 1
    assert pred_probs.shape[0] == y.shape[0]
    assert pred_probs.min() >= 0 and pred_probs.max() <= 1
    assert n_bins > 1
    bins = np.linspace(0, 1, n_bins+1)
    binids = np.searchsorted(bins[1:-1], pred_probs)
    bin_sums = np.bincount(binids, weights=pred_probs, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / y.shape[0]))
    return ece


def test(pred_probs, y):
    assert pred_probs.ndim == 2 and y.ndim == 1
    y_onehot = np.eye(pred_probs.shape[1])[y]
    n_classes = pred_probs.shape[1]
    return {
        # 'ece': calc_ece(pred_probs, y, n_bins=int(y.shape[0]**0.5)),
        'accuracy': (np.argmax(pred_probs, axis=-1) == y).mean(),
        # 'nll': -np.mean(np.log(pred_probs)*y + np.log(1-pred_probs)*(1-y)),
        'nll': -np.mean(np.log((pred_probs*y_onehot).sum(-1)))
    }


def get_predictor(args, n_classes) -> Predictor:
    if args.method.startswith('ag'):
        from ag_predictors import AGBasePredictor, AGIPBPredictor
        if args.method == 'ag':
            return AGBasePredictor(args.ag_preset, args.n_threads)
        ipb_ntot_ratio = {
            'ag-ipb': args.ipb_N_ratio,
            'ag-bagging': 0
        }[args.method]
        return AGIPBPredictor(
            args.ag_preset, args.n_threads, args.ens_size, args.ipb_N_each, ipb_ntot_ratio, args.train_seed)

    tree_params = {
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators,
        'learning_rate': args.lr,
        'objective': 'multi:softprob', # 'binary:logistic',
        'num_class': n_classes,
        'nthread': args.n_threads,
        'seed': args.train_seed,
    }
    if args.method in ['tree', 'bagging']:
        if args.method == 'tree':
            return TreeEnsemble(tree_params, 1, 1)
        else:
            return TreeEnsemble(tree_params, args.ens_size, args.bagging_ratio)
    elif args.method == 'ipb':
        return TreeIPB(tree_params, args.ens_size, args.ipb_N_ratio, args.ipb_N_each)

    raise ValueError(f'unknown method: {args.method}')


def main(args):
    Dtr, Dva, Dtest = datasets.load(args.dataset, args.data_seed)
    assert Dtr[1].ndim == 1 and arr_is_int(Dtr[1]) and Dtr[1].min() == 0, ValueError(
        f"dataset must be classification: {Dtr[1].shape}, {Dtr[1].dtype}, {Dtr[1].min()}")
    n_classes = Dtr[1].max() + 1
    print('n_classes:', n_classes)
    if args.subsample_n > 0:
        Dtr = (Dtr[0][:args.subsample_n], Dtr[1][:args.subsample_n])

    predictor = get_predictor(args, n_classes)
    predictor.fit(Dtr, Dva)

    to_dump = {}
    for pref, d in [('val', Dva), ('test', Dtest)]:
        ypred = predictor.predict_proba(d[0])  # [n_test, n_classes]
        res = test(ypred, d[1])
        to_dump.update({f'{pref}_{k}': v for k, v in res.items()})
    print(to_dump)

    if not args.method.startswith('ag'):
        # add feature importance scores
        to_dump = {
            'fscore': predictor.get_fscores()
        }

    with open(os.path.join(args.dir, 'dump.pkl'), 'wb') as fout:
        pickle.dump(to_dump, fout)


if __name__ == '__main__':
    args = get_parser().parse_args()
    exputils.preflight(args)
    main(args) 
