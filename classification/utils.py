import numpy as np
import pandas as pd
from typing import Tuple


DTuple = Tuple[pd.DataFrame, np.ndarray]


def sample_categorical(p: np.ndarray, rng: np.random.Generator):
    return (p.cumsum(-1) >= rng.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def arr_is_int(a: np.ndarray):
    return np.issubdtype(a.dtype, np.integer)


def polya_urn_sample(n: int, delt_n: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.arange(n)
    for _ in range(delt_n):
        idx = rng.integers(0, pool.shape[0])[None]
        pool = np.concatenate([pool, pool[idx]], 0)
    return pool[n:]


def subsample_dtuple(Dtup: DTuple, ratio: float, rng: np.random.Generator) -> DTuple:
    if ratio == 1:
        return Dtup

    if ratio == -1:  # with replacement, for Efron's bootstrap
        idcs = rng.choice(Dtup[0].shape[0], Dtup[0].shape[0], replace=True)
    else:  # w/o replacement for standard bagging
        idcs = rng.choice(Dtup[0].shape[0], int(Dtup[0].shape[0]*ratio), replace=False)

    if isinstance(Dtup[0], np.ndarray):
        return Dtup[0][idcs], Dtup[1][idcs]
    else:
        return Dtup[0].iloc[idcs], Dtup[1][idcs]