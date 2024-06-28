from collections import namedtuple
import logging, sys, socket
import os, json, hashlib
logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

import runner


class Object(object):

    def __init__(self, **kw):
        vars(self).update(kw)


def split_args(a, typ=int):
    return [] if a=='' else list(map(typ, a.split(',')))


_F = namedtuple('_F', 'k v is_pref')
_BF = namedtuple('_BF', 'v')
def F(k, *vs): return [_F(k, v, False) for v in vs]
def PF(pref, *vs): return [_F(pref, v, True) for v in vs]
def Bool(no_first=False): return [_BF(False), _BF(True)] if no_first else [_BF(True), _BF(False)]


CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
Info = namedtuple('Info', 'root_cmd log_dir_base cwd sep', defaults=[CWD, '--'])


def proc(locals_dict):
    kv: Info = locals_dict['__info']

    opts = {}

    cmd_suff = ''
    for k, v in locals_dict.items():
        if k.startswith('__'): continue
        if isinstance(v, _F):
            if v.is_pref:
                k = v.k + '.' + k
            else:
                k = v.k
            v = v.v
        elif isinstance(v, _BF):
            if not v.v:
                k = 'no_' + k
            v = ''
        opts[k] = v
        cmd_suff += f' {kv.sep}{k}'
        if isinstance(v, list):
            v = ' '.join(map(str, v))
        if v != '':
            cmd_suff +=f' {v}'

    # path length limit of modern FS is 256B; this is 56B
    exp_dir = hashlib.sha224(cmd_suff.encode('utf-8')).hexdigest()
    log_dir = os.path.join(kv.log_dir_base, exp_dir)
    cmd = kv.root_cmd + f' -dir={log_dir}' + cmd_suff

    rem = dict((k, None) for k in ['post_cmd', 'group_id', 'max_cpu_time', 'post_kill_cmd'])
    return runner.Task(
        cmd=cmd, option_dict=opts, work_dir=kv.cwd, log_dir=log_dir, ttl=0, **rem)