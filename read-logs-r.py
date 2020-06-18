#!/usr/bin/env python3
""" Read/process the logs produced during tuning trac models.
    Default prints out the best score, -t argument produces a table. 
"""

import json
import sys
import numpy as np
import argparse
from collections import OrderedDict # used as an `ordered' set

def read_logs(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            if (len(line) > 1 and line[0] != '#'):
                log_data = json.loads(line)
                tuned_params = log_data['params']
                model_params = log_data['defaults']
                scores = log_data['scores']

                yield tuned_params, scores, model_params

ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('logfiles', nargs="+", help='Log file(s).')
ap.add_argument('--all-params', '-a', action='store_true',
        help='Output all model parameters, including defaults.')
ap.add_argument('--optimize', '-o',
        default='r2',
        help='The score to optimize (ignored if -t is given)')
grp = ap.add_mutually_exclusive_group()
grp.add_argument('--tsv', '-t', action='store_true',
        help='Output a tab-seperated file table.')
grp.add_argument('--top-n', '-T', default=20, type=int,
        help='Only output records with TOP-N F1-scores.')
grp.add_argument('--ngram-table', '-n', action='store_true',
        help='Output a table with best scores for each char/word n-gram cobination.')
opt = ap.parse_args()

scores = []     # scores
tunedp = []     # parameters used during tuning
params = []     # all model parameters, including defaults
for f in opt.logfiles:
    for i, (tp, sc, mp) in enumerate(read_logs(f)):
        scores.append(sc)
        params.append(mp)
        tunedp.append(tp)

if opt.optimize in sc:
    optimize = opt.optimize
else:
    optimize = None
    for x in ('f1-score:binary', 'f1-score:macro', 'r2'):
        if x in sc:
            optimize = x

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
        print(fmt.format(*[out.get(x, 'NA') for x in col_names]))
elif opt.ngram_table:
    ngram_dict = dict()
    cmin, cmax, wmin, wmax = 100, 0, 100, 0
    for i, (tp, sc, mp) in enumerate(zip(tunedp,scores,params)):
        sc_mean = np.mean(sc[optimize]) 
        if sc_mean > ngram_dict.get((tp['c_ngmax'], tp['w_ngmax']), float('-inf')):
            ngram_dict[(tp['c_ngmax'], tp['w_ngmax'])] = sc_mean
        if cmin > tp['c_ngmax']: cmin = tp['c_ngmax']
        if cmax < tp['c_ngmax']: cmax = tp['c_ngmax']
        if wmin > tp['w_ngmax']: wmin = tp['w_ngmax']
        if wmax < tp['w_ngmax']: wmax = tp['w_ngmax']
    print(cmin, cmax, wmin, wmax)
    print(ngram_dict)
    print("" + ((wmax-wmin+1)*"\t{}").format(*list(range(wmin, wmax+1))))
    for cng in range(cmin, cmax+1):
        print("{}".format(cng), end="")
        for wng in range(wmin, wmax+1):
            print("\t{}".format(ngram_dict.get((cng,wng), '')), end="")
        print()
else:
    best_i = 0      # index of best score
    best_mean = 0.0
    n = 0
    for i, (tp, sc, _) in enumerate(zip(tunedp,scores,params)):
        if optimize is None:
            optmize = sc.keys()[0]
        sc_mean = np.mean(sc[optimize]) 
        if sc_mean > best_mean:
            best_mean = sc_mean
            best_i = i
        n += 1

    best_sc = scores[best_i]
    sc_names = best_sc.keys()
    best_params = tunedp[best_i]
    fmt = len(sc_names) * "{}: {:0.4f}±{:0.4f} "
    best_sc = [(k, np.mean(best_sc[k]), np.std(best_sc[k]))
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
        print("{:0.4f}±{:0.4f} ".format(
            np.mean(sc[opt.optimize]), np.std(sc[opt.optimize])),
            end="")
        print_end = ","
        for i, k in enumerate(sorted(param)):
            v = param[k]
            if i == (len(param) - 1):
                print_end = ""
            if isinstance(v, float):
                print('{}={:0.4f}'.format(k,v), end=print_end)
            else:
                print('{}={}'.format(k,v), end=print_end)
        print()
