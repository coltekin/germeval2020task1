#!/usr/bin/env python3

import logging
from logging import debug, info, warning, basicConfig
basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

import argparse
import sys
import os
import csv
import json
import random
from hashlib import md5
import re
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import \
        ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import rankdata, pearsonr, describe
from scipy.sparse import hstack

from task1data import Task1data
from tune import tune, read_logs



tokenize = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall
def get_ngrams(s, ngmin, ngmax, separator="",
               bos="<", eos=">", suffix="", flatten=True):
    """ For the given sequence s. Return all ngrams in range ngmin-ngmax.
        spearator is useful for readability
        bos/eos symbols are added to indicate beginning and end of seqence
        suffix is an arbitrary string useful for distinguishing
                in case differernt types of ngrams are used
        if flatten is false, a list of lists is returned where the
                first element contains the ngrams of size ngmin
                and the last contains the ngrams of size ngmax
    """

    # return a single dummy feature if there are no applicable ngrams
    # probably resulting in a mojority-class classifier
    if ngmax == 0 or (ngmax - ngmin < 0) :
        return ['__dummy__']

    ngrams = [[] for x in range(1, ngmax + 1)]
    s = [bos] + s + [eos]
    for i, ch in enumerate(s):
        for ngsize in range(ngmin, ngmax + 1):
            if (i + ngsize) <= len(s):
                ngrams[ngsize - 1].append(
                        separator.join(s[i:i+ngsize]) + suffix)
    if flatten:
        ngrams = [ng for nglist in ngrams for ng in nglist]
    return ngrams


def doc_analyzer(doc, lowercase='word',
        c_ngmin=1, c_ngmax=1,
        w_ngmin=1, w_ngmax=1):
    """ Convert document to word/char ngrams with optional
        case normaliztion.
    """

    if lowercase in {'both', 'all'}:
        lowercase = {'char', 'word'}
    else: lowercase = {lowercase}

    # character n-grams
    if 'char' in lowercase:
        docfeat = get_ngrams(list(doc.lower()),
                c_ngmin, c_ngmax)
    else:
        docfeat = get_ngrams(list(doc),
                c_ngmin, c_ngmax)
    # word n-grams
    if 'word' in lowercase:
        docfeat.extend(get_ngrams(tokenize(doc.lower()),
                                w_ngmin, w_ngmax,
                                suffix="⅏", separator=" "))
    else:
        docfeat.extend(get_ngrams(tokenize(doc),
                                w_ngmin, w_ngmax,
                                suffix="⅏", separator=" "))
    return docfeat

class DocAnalyzer:
    """ Just a trick to make the vectorizer pickleable.
    """
    def __init__(self, lowercase='word', 
                 c_ngmin=1, c_ngmax=1, w_ngmin=1, w_ngmax=1): 
        self.lowercase = lowercase
        self.c_ngmin = c_ngmin
        self.c_ngmax = c_ngmax
        self.w_ngmin = w_ngmin
        self.w_ngmax = w_ngmax
    def __call__(self, doc):
        return doc_analyzer(doc,
                    c_ngmin=self.c_ngmin,
                    c_ngmax=self.c_ngmax,
                    w_ngmin=self.w_ngmin,
                    w_ngmax=self.w_ngmax,
                    lowercase=self.lowercase)

class TextRegressor:
    param_defaults = {'min_df': 1, 'c_ngmin': 1, 'c_ngmax': 1,
                      'w_ngmax': 1, 'w_ngmin': 1, 'lowercase': 'word',
                      'alpha': 1.0, 'C': 1.0, 'mix': 1.0}
    def __init__(self, regressor='ridge', vectorizer='tf-idf'):
        if regressor == 'ridge':
            from sklearn.linear_model import Ridge
            self.reg = Ridge()
        elif regressor == 'SVR':
            from sklearn.svm import SVR
            self.reg = SVR()
        elif regressor == 'linearsvr':
            from sklearn.svm import LinearSVR
            self.reg = LinearSVR()
        if vectorizer == 'tf-idf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer()
        self.vec_params_default = self.vec.get_params()
        self.reg_params_default = self.reg.get_params()
        self._reset()

    def _reset(self):
        self.par = dict(self.param_defaults)
        self.vec_params = self.vec_params_default
        self.vec.set_params(**self.vec_params)
        self.reg_params = self.reg_params_default
        self.reg.set_params(**self.reg_params)

    def set_params(self, **params):
        self._reset()
        self.par.update(params)
        ngram_analyzer = DocAnalyzer(
                    lowercase=self.par.get('lowercase'),
                    c_ngmin=self.par.get('c_ngmin'),
                    c_ngmax=self.par.get('c_ngmax'),
                    w_ngmin=self.par.get('w_ngmin'),
                    w_ngmax=self.par.get('w_ngmax'))
        self.vec_params.update(
            {k:self.par[k] for k in self.par.keys() & self.vec_params.keys()})
        self.vec.set_params(**self.vec_params)
        self.vec.set_params(analyzer=ngram_analyzer)
        self.reg_params.update(
            {k:self.par[k] for k in self.par.keys() & self.reg_params.keys()})
        self.reg.set_params(**self.reg_params)

    def get_params(self):
        return self.par

    def fit(self, text, outcome):
        num = None
        if len(text) == 2:
            text, num = text
        x = self.vec.fit_transform(text)
        if num is not None:
            x = hstack((x, self.par['mix'] * num), format='csr')
        self.reg.fit(x, outcome)

    def predict(self, text,
                gold=None, gold_rank=None, rank_dir=-1, return_score=False):
        num = None
        if len(text) == 2:
            text, num = text
        x = self.vec.transform(text)
        if num is not None:
            x = hstack((x, self.par['mix'] * num), format='csr')
        pred = self.reg.predict(x)
        if return_score:
            return pred, self._score(gold, pred, gold_rank, rank_dir)
        else:
            return pred

    def _score(self, gold, pred, gold_rank=None, rank_dir=-1,
            verbose=False):
        r2 = r2_score(gold, pred)
        rmse = np.sqrt(mean_squared_error(gold, pred))
        if gold_rank is None:
            gold_rank = rankdata(rank_dir * gold, method='ordinal')
        pred_rank = rankdata(rank_dir * pred, method='ordinal')
        corr, _ = pearsonr(gold, pred)
        rank_corr, _ = pearsonr(gold_rank, pred_rank)
        if verbose:
            fmt = ("{}: n={}, min={:.4f}, max={:.4f}, mean={:.4f}, "
                   "var={:.4f}, skew={:.4f}, kurtosis={:.4f}")
            gold_dsc = describe(gold)
            pred_dsc = describe(pred)
            print(fmt.format('gold',
                gold_dsc[0], *gold_dsc[1], *gold_dsc[2:]))
            print(fmt.format('pred',
                pred_dsc[0], *pred_dsc[1], *pred_dsc[2:]))
        return {'r2': r2, 'rmse': rmse, 'rank_corr': rank_corr, 'corr': corr}

    def score(self, text, gold, gold_rank=None, rank_dir=-1,
            verbose=False):
        pred = self.predict(text)
        return self._score(gold, pred, gold_rank, rank_dir,
                verbose=verbose)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('tune', 'predict'))
    ap.add_argument('outcome', nargs="+")
    ap.add_argument('--params', '-p')
    ap.add_argument('--normalize-scores', '-Z', action='store_true')
    ap.add_argument('--regressor', '-r',
            choices=('ridge', 'linearsvr', 'svr'), default='ridge')
    ap.add_argument('--out_file', '-o')
    ap.add_argument('--n-procs', '-j', default=2, type=int)
    ap.add_argument('--log-prefix', '-S')
    ap.add_argument('--test-data', '-t', action='store_true', help="produce predictions on the test data.")
    ap.add_argument('--max-iter', '-M', type=int, default=1000)
    ap.add_argument('--aggregate', '-A', action='store_true')
    ap.add_argument('--top-ensemble', '-T', type=int, default=1)
    ap.add_argument('--optimize', '-O', default='r2')
    ap.add_argument('--extra-features', '-x', action='store_true')
    args = ap.parse_args()

    trn = Task1data()
    trn.load('train', normalize=args.normalize_scores)
    if args.command == 'tune':
        trn.load('dev', normalize=args.normalize_scores)
        trn_text, trn_out, trn_stuids = trn.get_data2(args.outcome + ['rank'],
                        aggregate=args.aggregate, return_extra=args.extra_features)
        logfile = sys.stdout
        for i, outcome in enumerate(args.outcome):
            if args.log_prefix:
                logfile = open(
                        "{}-{}.log".format(args.log_prefix, outcome), "at")
            trn_data = (trn_text, trn_out[:, i])
            if args.extra_features:
                trn_data = (trn_text[0], trn_out[:, i], trn_text[1])
            tune(TextRegressor, args.params, trn_data,
                    init_args={'regressor': args.regressor}, save=logfile,
                    k=5, optimize='rank_corr', max_iter=args.max_iter)
            if args.log_prefix: logfile.close()

    if args.command == 'predict':
        if args.test_data:
            trn.load('dev', normalize=args.normalize_scores)
            tst = Task1data()
            tst.load('test', normalize=args.normalize_scores)
        else:
            tst = Task1data()
            tst.load('dev', normalize=args.normalize_scores)

        tst_text, tst_out, tst_stuids = tst.get_data2(
                args.outcome + ['rank'], aggregate=args.aggregate,
                    return_extra=args.extra_features) 
        trn_text, trn_out, trn_stuids = trn.get_data2(
                args.outcome + ['rank'], aggregate=args.aggregate,
                return_extra=args.extra_features)
        if not args.test_data:
            tst_rank = tst_out[:, len(args.outcome)].ravel()
        predictions = list()
        for i, outcome in enumerate(args.outcome):
            if args.log_prefix:
                logfile = open(
                        "{}-{}.log".format(args.log_prefix, outcome), "at")
            nbest = [(float("-inf"), None) for _ in range(args.top_ensemble)]
            for p, sc in read_logs("{}-{}.log".format(
                        args.log_prefix, outcome)):
                mean_sc = np.mean(sc[args.optimize]) 
                if mean_sc > nbest[-1][0]:
                    nbest.pop()
                    nbest.append((mean_sc, p))
                    nbest = sorted(nbest, key=lambda x: x[0], reverse=True)
#            print(outcome, best_p, best_sc)
            pred_ensemble = list()
            for j, (sc, p) in enumerate(nbest):
                m = TextRegressor(regressor=args.regressor)
                m.set_params(**p)
                m.fit(trn_text, trn_out[:, i])
                if args.test_data:
                    pred = m.predict(tst_text)
                else:
                    pred, score = m.predict(tst_text, tst_out[:, i],
                            gold_rank=tst_rank, return_score=True)
                    print("{}[{}]: {}".format(outcome, j, score))
                pred_ensemble.append(pred)
            pred_ensemble = np.array(pred_ensemble).mean(axis=0).ravel()
            if not args.test_data:
                sc = m._score(tst_out[:, i], pred_ensemble, gold_rank=tst_rank)
                print("{} {}".format(outcome,sc))
            predictions.append(pred_ensemble)
        predictions = np.array(predictions).T
        pred_sum = predictions.sum(axis=1).ravel()
        if not args.test_data:
            gold_sum = tst_out[:,:-1].sum(axis=1).ravel()   # *_out[-1] is rank
            sc = m._score(gold_sum, pred_sum, gold_rank=tst_rank, verbose=True)
            print("aggr {}".format(sc))
        if not args.aggregate:
            stuid_i = {k:i for i,k in enumerate(tst.data.keys())}
            pred_sum_aggr = [list() for _ in stuid_i]
            if not args.test_data:
                gold_sum_aggr = [None for _ in stuid_i]
            for i, stuid in enumerate(tst_stuids):
                pred_sum_aggr[stuid_i[stuid]].append(pred_sum[i])
                if not args.test_data:
                    gold_sum_aggr[stuid_i[stuid]] = gold_sum[i]
            pred_sum = np.array(pred_sum_aggr).mean(axis=1).ravel()
            if not args.test_data:
                gold_sum = np.array(gold_sum_aggr)
                gold_rank = rankdata(-1*gold_sum, method='ordinal')
                sc = m._score(gold_sum, pred_sum, gold_rank=gold_rank,
                        verbose=True)
                print("aggr2 {}".format(sc))
        if args.test_data and not args.out_file:
            args.out_file = "predictions.txt"
        if args.out_file:
            pred_rank = rankdata(-1*pred_sum, method='ordinal')
            with open(args.out_file, "wt") as fp:
                print("task1", file=fp)
                for i, stuid in enumerate(tst.data.keys()):
                    print("{}\t{}".format(stuid, pred_rank[i]), file=fp)
