import monad_do as M
import runner
from runner.utils import _get_timestr
import os
import shutil
import sys
import argparse
import pandas as pd

try:
    from .utils import *
except:
    from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_max_gpus', '-ng', type=int, default=1)
parser.add_argument('--n_multiplex', '-nm', type=int, default=2)
parser.add_argument('--seed_s', '-ss', type=int, default=5)
parser.add_argument('--seed_e', '-se', type=int, default=15)
parser.add_argument('-rmk', type=str, default='')
parser.add_argument('-train', action='store_true', default=False)
parser.add_argument('-rlog', type=str, default='')
parser.add_argument('--dir_prefix', '-dp', type=str, default=os.path.expanduser('~/run/itest'))
parser.add_argument('--min_storage_per_task', type=float, default=5)
parser.add_argument('--perf_path', '-P', type=str)


args = parser.parse_args()


DATA_KEYS = ['dataset']
HPS_KEYS = ['max_depth', 'n_estimators', 'lr']

def load_best_hps():
    df = pd.read_csv(args.perf_path)
    SEED = 'data_seed'
    RES_KEY = 'val_nll'
    ALL_KEYS = HPS_KEYS + DATA_KEYS + [SEED] + [RES_KEY]
    gb = df.groupby(DATA_KEYS + HPS_KEYS).mean().groupby(DATA_KEYS)
    return list(gb[RES_KEY].idxmin().values)


@M.do(M.List)
def list_hps(__info):

    data_seed = yield list(range(args.seed_s, args.seed_e))
    train_seed = data_seed 

    __hps = yield load_best_hps()

    dataset = __hps[0]
    max_depth = __hps[1]
    n_estimators = __hps[2]
    lr = __hps[3]
    method = yield ['tree', 'bagging', 'ipb']

    if method == 'ipb':
        ipb_N_ratio = yield [1] 
        ipb_N_each = yield [0.125, 0.25, 1]

    return [proc(locals())]


remark = os.path.basename(__file__).split('.')[0] + '_' + args.rmk
exp_dir_base = args.dir_prefix
log_dir_base = os.path.join(exp_dir_base, f'{_get_timestr()}_{remark}')
env_pref = f'CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_MEM_FRACTION={0.8/args.n_multiplex:.3f} OMP_NUM_THREADS=2 TQDM_MININTERVAL=10 '
root_cmd = env_pref + 'python main.py -production --ens_size 50 '
cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../classification')
tasks = list_hps(Info(root_cmd=root_cmd, log_dir_base=log_dir_base, cwd=cwd))


print('\n'.join([t.cmd for t in tasks[-100:]]))
print(len(tasks))
free_space_GB = shutil.disk_usage(exp_dir_base).free / 1e9  # raises error if not exist
required_GB = args.min_storage_per_task * len(tasks) / 1e3  # MB -> GB

print(f'storage: {free_space_GB} free, {required_GB} requested')
if required_GB >= free_space_GB:
    print(f'insufficient storage')
    sys.exit(1)

if not args.train:
    sys.exit(0)

os.makedirs(log_dir_base, exist_ok=True)
shutil.copyfile(__file__, os.path.join(log_dir_base, 'script.py'))
with open(os.path.join(log_dir_base, 'script.py'), 'a') as fout:
    print('#', ' '.join(sys.argv), file=fout)

with open(os.path.join(log_dir_base, 'NAME'), 'w') as fout:
    print(args.rmk, file=fout)

ng = None if args.n_max_gpus == -1 else args.n_max_gpus  # all
r = runner.Runner(
    n_devices=ng,
    n_multiplex=args.n_multiplex,
    n_max_retry=-1,
    require_gpu=False,
    log_url=args.rlog)
r.run_tasks(tasks)

with open(os.path.join(log_dir_base, 'COMPLETE'), 'w') as fout:
    fout.write('.')
