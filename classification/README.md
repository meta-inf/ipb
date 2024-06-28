This directory contains code to reproduce the experiments for the classification experiments.

## Using the Code

Download the pre-processed datasets from [this link](https://drive.google.com/file/d/19bjIrqD0RUJ1pliwSQemYf8LeRMd3rwh/view?usp=sharing) 
and extract them to `data/`. 

Then, to reproduce the OpenML experiments:

```sh
conda create -n ag python=3.10 pip
conda activate ag 
pip install -r requirements-ag.txt   # replace ag->xgb for the XGBoost experiments

N_PROCS=24  # NOTE replace this
ROOT_PATH=~/run/itest
mkdir -p $ROOT_PATH
cd ../scripts
python ag.py -ng 1 -nm ${N_PROCS} -dp ${ROOT_PATH} -train
python xgb.py -P cc18-val.csv -ng 1 -nm ${N_PROCS} -dp ${ROOT_PATH} -train
```

You should make sure the server has at least `$N_PROCS * 4` cores.

Upon completion there will be two directories inside `$ROOT_PATH`.  The following commands will
reproduce Table 4, 6, 7 and Figure 3 and save them to the current directory.

```sh
python gather.py $ROOT_PATH/$AG_EXP_DIR  ag.csv
python gather.py $ROOT_PATH/$XGB_EXP_DIR xgb.csv
python proc-results.py xgb.csv ag.csv
```

Samples of the CSV files are provided in `scripts/`.

Figure 4 is obtained using the following run (the hyperparameters are similarly determined by
validation loss): 

```sh
python main.py --dataset adult --max_depth 7 --n_estimators 200 --lr 0.15 --ens_size 400 -dir $OUT_DIR
```

which exports the feature importance scores to `$OUT_DIR/dump.pkl`.