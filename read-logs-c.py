#!/usr/bin/env python3
""" Read/process the logs produced during tuning trac models.
    Default prints out the best score, -t argument produces a table. 
"""

import json
import sys
import numpy as np
import argparse
from collections import OrderedDict # used as an `ordered' set
from textc import read_logs

ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('logfiles', nargs="+", help='Log file(s).')
ap.add_argument('--all-params', '-a', action='store_true',
        help='Output all model parameters, including defaults.')
grp = ap.add_mutually_exclusive_group()
grp.add_argument('--tsv', '-t', action='store_true',
        help='Output a tab-seperated file table.')
grp.add_argument('--top-n', '-T', default=20, type=int,
        help='Only output records with TOP-N F1-scores.')
grp.add_argument('--optimize', '-o',
        default='f1-score:macro',
        help='The score to optimize (ignored if -t is given)')
opt = ap.parse_args()

scores = []     # scores
tunedp = []     # parameters used during tuning
params = []     # all model parameters, including defaults
for f in opt.logfiles:
    for i, (tp, sc, mp) in enumerate(read_logs(f)):
        scores.append(sc)
        params.append(mp)
        tunedp.append(tp)

if opt.tsv:
    for i, (tp, sc, mp) in enumerate(zip(tunedp,scores,params)):
        if i == 0:
            col_names = sorted(tp.keys())
            col_names += [sc + "_mean" for sc in sorted(sc.keys())]
            col_names += [sc + "_sd" for sc in sorted(sc.keys())]
            if opt.all_params:
                col_names += sorted(mp.keys())
            fmt = "{}" + "\t{}" * (len(col_names) - 1)
            print(fmt.format(*list(col_names)))

        out = tp
        sc_mean = {x+'_mean':np.mean(sc[x]) for x in sc.keys()}
        sc_sd = {x+'_sd':np.std(sc[x]) for x in sc.keys()}
        out.update(sc_mean)
        out.update(sc_sd)
        if opt.all_params: out.update(mp)
        print(fmt.format(*[out[x] for x in col_names]))
else:
    best_i = 0      # index of best score
    best_mean = 0.0
    n = 0
    for i, (tp, sc, _) in enumerate(zip(tunedp,scores,params)):
        if opt.optimize in sc:
            optimize = opt.optimize
        else:
            optimize = None
            for x in ('f1-score:binary', 'f1-score:micro'):
                if x in sc:
                    optimize = x
            if optimize is None:
                optmize = sc.keys()[0]
        sc_mean = np.mean(sc[opt.optimize]) 
        if sc_mean > best_mean:
            best_mean = sc_mean
            best_i = i
        n += 1


    best_sc = scores[best_i]
    sc_names = best_sc.keys()
    best_params = tunedp[best_i]
    fmt = len(sc_names) * "{}: {:0.4f}±{:0.4f} "
    best_sc = [(k, 100*np.mean(best_sc[k]), 100*np.std(best_sc[k]))
            for k in sorted(best_sc)]
    best_sc = [x for t in best_sc for x in t]
    print('Based on {} entries.'.format(n))
    print('Best score:', fmt.format(*best_sc))
    print('Top {}:'.format(opt.top_n))
    if opt.all_params:
        sc_param = zip(scores, params)
    else:
        sc_param = zip(scores, tunedp)
    for sc, param  in \
       sorted(sc_param, 
            key=lambda x: np.mean(x[0][opt.optimize]), reverse=True)[:opt.top_n]:
        print("{:0.2f}±{:0.2f} ".format(
            100*np.mean(sc[opt.optimize]), 100*np.std(sc[opt.optimize])),
            end="")
        print_end = ","
        for i, k in enumerate(sorted(param)):
            v = param[k]
            if i == (len(param) - 1):
                print_end = ""
            if isinstance(v, float):
                print('{}={:0.2f}'.format(k,v), end=print_end)
            else:
                print('{}={}'.format(k,v), end=print_end)
        print()
