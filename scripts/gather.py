import os, sys, json, numpy as np, os.path as osp, pandas as pd

def load(base_path):
    nan = np.nan
    inf = np.inf

    recs = []

    for p in os.listdir(base_path):
        if not osp.exists(osp.join(base_path, p, 'hps.txt')):
            continue
        with open(osp.join(base_path, p, 'hps.txt')) as fin:
            hps = json.load(fin)
        try:
            with open(osp.join(base_path, p, 'stdout')) as fin:
                line = eval(fin.readlines()[-1])
            recs.append(hps | line)
        except Exception as _:
            print(hps, 'crashed')
            continue

    df = pd.DataFrame(recs)
    return df

load(sys.argv[1]).to_csv(sys.argv[2])
