import pandas as pd, numpy as np, os, sys, os.path as osp, json
from scipy.stats import wilcoxon, ttest_rel

from matplotlib import pyplot as plt


def ref_line(**kw):
    xmn, xmx = plt.xlim()
    ymn, ymx = plt.xlim()
    mn, mx = min(xmn, ymn), max(xmx, ymx)
    plt.plot([mn, mx], [mn, mx], linestyle=':', c='gray', **kw)
    plt.xlim(mn/1.001, mx*1.001)
    plt.ylim(mn/1.001, mx*1.001)
    

def wilcoxon_pval(results, baselines):
    return wilcoxon(results - baselines, alternative='less')[1]

    
def has_significant_improvement(results, baselines):
    # results / baseline_results are loss values, lower is better
    diff = results - baselines
    if np.max(np.abs(diff)) < 1e-5:
        return False
    return wilcoxon_pval(results, baselines)
    

def finite_mean(a):
    # AutoML baselines occasionally produce infty test loss
    a = np.asarray(a)
    return a[~np.isinf(a)].mean()


def finite_std(a):
    a = np.asarray(a)
    return a[~np.isinf(a)].std()


cc18_data_names = ['banknote-authentication', 'blood-transfusion-service-center', 'balance-scale', 'dresses-sales', 'mfeat-fourier', 'MiceProtein', 'steel-plates-fault', 'climate-model-simulation-crashes', 'car', 'cylinder-bands', 'breast-w', 'mfeat-karhunen', 'mfeat-morphological', 'eucalyptus', 'mfeat-zernike', 'cmc', 'credit-approval', 'vowel', 'credit-g', 'analcatdata_authorship', 'analcatdata_dmft', 'diabetes', 'pc4', 'pc3', 'kc2', 'pc1', 'tic-tac-toe', 'vehicle', 'wdbc', 'qsar-biodeg', 'ilpd']

DATA_KEYS = ['dataset', 'subsample_n']
METHOD_KEYS = ['method']
IPB_HPS = ['ipb_N_each', 'ipb_N_ratio']
SEED = 'data_seed'
RES_KEYS = ['val_nll', 'val_accuracy', 'test_nll', 'test_accuracy']

# ============= LOAD XGB results ============= 
df = pd.read_csv(sys.argv[1])
df.sort_values(by='data_seed', inplace=True)
df1 = df[df.method!='ipb'].filter(
    DATA_KEYS+METHOD_KEYS+RES_KEYS+['data_seed']
).groupby(DATA_KEYS+METHOD_KEYS).agg(list).reset_index()
# for the proposed method we use val nll to select the sampling scheme
df2 = df[df.method=='ipb']
df2g = df2.filter(
    DATA_KEYS+METHOD_KEYS+IPB_HPS+RES_KEYS+['data_seed']
).groupby(
    DATA_KEYS+METHOD_KEYS+IPB_HPS
).agg({
    k: ('mean' if k.startswith('val') else list)
    for k in RES_KEYS+['data_seed']
})
df2_grouped = df2g.reset_index()
idcs = df2_grouped.groupby(
    DATA_KEYS+METHOD_KEYS).val_nll.idxmin()
df2 = df2_grouped.loc[idcs].filter(DATA_KEYS+METHOD_KEYS+RES_KEYS+['data_seed'])
# 
df_xgb = pd.concat([df1, df2])
df_xgb.method = df_xgb.apply({'method': (lambda m: {
    'bagging': 'XGB+Bagging',
    'ipb': 'XGB+IPB',
    'tree': 'XGB'
}[m])})
df_xgb.sort_values(by='method', inplace=True)  # ['bagging', 'ipb', 'tree']

# ============= LOAD AutoGluon results ============= 
df_ag = pd.read_csv(sys.argv[2]).sort_values(by='data_seed').groupby(DATA_KEYS+METHOD_KEYS).agg(list).reset_index()
df_ag.method = df_ag.apply({
    'method': (lambda m: {
        'ag': 'AG',
        'ag-bagging': 'AG+Bagging',
        'ag-ipb': 'AG+IPB'
    }[m]),
})
df_ag.sort_values(by='method', inplace=True)
data_seeds = pd.concat([df_xgb.data_seed.apply(tuple), df_ag.data_seed.apply(tuple)])
assert len(data_seeds.unique()) == 1, "experiments must use matching seeds for the Wilcoxon test to be valid"
df = pd.concat([df_xgb, df_ag])
methods = df.method.unique()
dsets = sorted(df.dataset.unique())
boldface_best = False
n_methods = len(methods)
n_metrics = 2

# ================================================

cells = [[None for _ in range(n_methods * n_metrics)] for _ in dsets]
loss_by_metric_and_method = [
    [[_ for _ in dsets] for _ in range(n_methods)] for _ in range(n_metrics)]

for x, dset in enumerate(dsets):
    df_x = df[df.dataset==dset]
    assert tuple(df_x.method) == tuple(methods), dset
    y = 0
    for i_metric, (sgn, k) in enumerate([(1, 'test_nll'), (-1, 'test_accuracy')]):
        vals = list(df_x[k])
        vals = list(map(np.asarray, vals))
        for i, arr in enumerate(vals):
            best = True
            for j, arr_j in enumerate(vals):
                if i!=j and has_significant_improvement(sgn*arr_j, baselines=sgn*arr):
                    best = False
                    break
                    
            if k == 'test_nll':
                loss_sgn = 1
                mean = '{:.3f}'.format(finite_mean(arr))
                sd = '{:.2f}'.format(finite_std(arr) * 1.96 / arr.shape[0]**0.5)
                assert mean[0] == sd[0] == '0', (mean,sd)
                mean = mean[1:]
                sd = '$\\pm ' + sd[1:] + '$'
            else:
                loss_sgn = -100
                mean = '{:.1f}'.format(100*finite_mean(arr))
                sd = '$\\pm {:.1f}$'.format(100*finite_std(arr)* 1.96 / arr.shape[0]**0.5)

            loss_by_metric_and_method[i_metric][i][x] = loss_sgn * arr
            if not (best and boldface_best):
                cells[x][y] = '& $' + mean + r'${\tiny ' + sd + '}'
            else:
                cells[x][y] = r'& $\mathbf{' + mean + r'}${\tiny' + sd + '}'

            y = y+1
            
loss_by_metric_and_method = np.asarray(loss_by_metric_and_method)


# ================================= SCATTER PLOTS ================================= 

def PM(s):
    return s.replace('XGB', 'GDBT').replace('AG', 'AutoML')

for delt_y in [0, 3]:

    plt.figure(figsize=(7, 3), facecolor='w')
    plt.subplot(121)
    plt.scatter(-loss_by_metric_and_method[0, delt_y+0, :].mean(-1), 
                -loss_by_metric_and_method[0, delt_y+2, :].mean(-1), marker='+', s=40)
    plt.xlabel(PM(methods[delt_y+0])); plt.ylabel(PM(methods[delt_y+2])); plt.title('log likelihood')
    ref_line()

    plt.subplot(122)
    plt.scatter(-loss_by_metric_and_method[1, delt_y+0, :].mean(-1), 
                -loss_by_metric_and_method[1, delt_y+2, :].mean(-1), marker='+', s=40)
    plt.xlabel(PM(methods[delt_y+0])); plt.title('accuracy')
    ref_line()
    
    plt.tight_layout()

    plt.savefig(f'./cc18-ipb-vs-baseline-{delt_y}.pdf')


# ================================= RESULT TABLES ================================= 

def get_rank(loss_table, account_for_tie):
    n_methods, n_trials = loss_table.shape
    if not account_for_tie:
        avg = loss_table.mean(1)
        return avg.argsort().argsort() + 1
    ret = np.ones((n_methods,))
    for i in range(n_methods):
        for j in range(n_methods):
            if j != i:
                ret[i] += has_significant_improvement(loss_table[j], loss_table[i])
    return ret

def average_rank(loss_table, account_for_tie=False):
    """
    :param loss_table: [n_method, n_datasets, n_seeds]
    """
    sum_rank = np.zeros((n_methods, )).astype('f')
    loss_table = np.asarray(loss_table)
    for i_data in range(loss_table.shape[1]):
        sum_rank += get_rank(loss_table[:, i_data], account_for_tie)
    return sum_rank/loss_table.shape[1]

metrics = [
    ('NLL', 'NLL', '{:.3f}'.format), 
    ('accuracy', 'acc.', lambda a: '{:.1f}'.format(-a))
]
with open(f'./main-text-table.tex', 'w') as fout:  
    for i, (metric, metric_short, fmt_fn) in enumerate(metrics):
        print(metric, end='\t', file=fout)
        avg_rank = average_rank(loss_by_metric_and_method[i])
        avg_metrics = [finite_mean(m_j) for m_j in loss_by_metric_and_method[i]]
        for j in range(n_methods):
            j_s = j//3*3
            if avg_metrics[j] == min(avg_metrics[j_s:j_s+3]):
                mj = r'\mathbf{' + fmt_fn(avg_metrics[j]) + '}'
            else:
                mj = fmt_fn(avg_metrics[j])
            print('& ${}$'.format(mj), end='\t', file=fout)
        print(r'\\', file=fout)
        
        print(metric_short + ' rank', file=fout)
        for j in range(n_methods):
            rj = f'{avg_rank[j]:.2f}'
            j_s = j//3*3
            if avg_rank[j] == min(avg_rank[j_s:j_s+3]):
                metric = r'\mathbf{' + rj + '}'
            else:
                metric = rj
            print('& ${}$'.format(metric), end='\t', file=fout)
        print(r'\\', file=fout)
    
for d in [0,1]:
    with open(f'./appendix-table-cc18-{d}.tex', 'w') as fout:  
        print(r'''
        \begin{tabular}[h]{ccccccc} 
        \toprule 
        \multirow{2}{*}[-0.2em]{Method} & \multicolumn{3}{c}{GBDT} & \multicolumn{3}{c}{AutoML} \\ 
        \cmidrule(lr){2-4}  \cmidrule(lr){5-7}
            & (Base) & + BS & + IPB & (Base) & + BS & + IPB 
            \\ \midrule 
        '''.strip(), file=fout)

        for x in range(len(dsets)):
            task_rank = int(dsets[x].split('_')[1])
            dset_name = cc18_data_names[task_rank].replace('_', r'\_')
            print(dset_name, end='\t', file=fout)
            for y in range(len(methods)):
                # print(methods[y%len(methods)], end='\t')
                print(cells[x][y+n_methods*d], end='\t', file=fout)
            print(r'\\', file=fout)
            
        print(r'''\midrule 
        Wilcoxon p-value''', end='\t', file=fout)
        for y in range(n_methods):
            y_tgt = y//3*3+2
            if y == y_tgt:
                cell = '-'
            else:
                loss_tgt = loss_by_metric_and_method[d][y_tgt].mean(-1).reshape((-1,))
                loss_cur = loss_by_metric_and_method[d][y].mean(-1).reshape((-1,))
                cell = '{:.2g}'.format(wilcoxon_pval(loss_tgt, loss_cur))
            print('&', cell, end='\t', file=fout)
            
        print('\\\\\n', r'''\bottomrule
        \end{tabular}''', file=fout)
