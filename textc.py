#!/usr/bin/env python3
""" Base class(es) for text classifiers.

The file defines some of the common classes/functions as well as the
interface (including a command line interface).
"""

import sys, os, time, csv, re, itertools, random, json
import gzip
from hashlib import md5
import numpy as np
from collections import Counter, OrderedDict
import logging
from logging import debug, info, warning, basicConfig
basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler

_TOKENIZER = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall
_MAX_LEN = 1024*1024    # default maximum text length
_MIN_LEN = 0            # default minimum text length
def read_csv(path, header=None, sep='\t'):
    """ A generator for reading CSV files.

    Format is restricted to proper (quoted and escaped) CSV files 
    with a column for label and another for the text. The file may
    contain other fields, but only the class label and the text is
    used.

    Args:
        header: None or e sequence with two elements, specifying 
                the headers that correspond to label and
                the text respectively. If None, header is
                assumed to contain no headers.
    """
    if path.endswith('.gz'):
        import gzip 
        fp = gzip.open(path, 'rt')
    elif path.endswith('.xz'):
        import lzma 
        fp = lzma.open(path, 'rt')
    elif path.endswith('.bz2'):
        import bz2 
        fp = bz2.open(path, 'rt')
    else:
        fp = open(path, 'rt')
    if header is None:
        csvfp = csv.DictReader(fp, fieldnames=('label', 'text'),
                delimiter=sep)
        label_h, text_h = 'label', 'text'
    else:
        label_h, text_h = header
        csvfp = csv.DictReader(fp, delimiter=sep)
    for row in csvfp:
        yield row[label_h], row[text_h]
    fp.close()

def get_ngrams(s, ngmin, ngmax, separator="",
               bos="<", eos=">", suffix="", flatten=True):
    """ For the given sequence s. Return all ngrams in range ngmin-ngmax.
        spearator is useful for readability
        bos/eos symbols are added to indicate beginning and end of seqence
        suffix is an arbitrary string useful for distinguishing
                in case differernt types of ngrams
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

class TextCData(object):
    def __init__(self, path=None,
            num_data=None,
            cat_data=None,
            tokenizer=_TOKENIZER,
            negative_class=None,
            labels=[],
            maxlen=_MAX_LEN, minlen=_MIN_LEN,
            text_label="text",
            class_label="label",
            sep='\t'):
        # _id is a dirty hack to identify the objec quickly
        self._id = random.getrandbits(64) ^ int(10000*time.time())
        self.maxlen = maxlen
        self.text_label = text_label
        self.class_label = class_label
        self.delimiter = sep
        self.minlen = minlen
        self.texts = []
        self.labels = []
        self.num_data = num_data
        self.cat_data = cat_data
        self.num_features = None
        self.cat_features = None
        self.label_names = OrderedDict()
        self.negative_class = negative_class
        if negative_class:
            self.label_names[negative_class] = 0
        for l in labels:
            if l not in self.labels:
                self.label_names[l] = len(self.label_names)
        self.tokenizer = tokenizer
        if path:
            self.load(path, num_data, cat_data)
    def __eq__(self, other):
        if isinstance(other, TextCData):
            return self._id == other._id
        else:
            return False

    def symbolic_labels(self, index=None):
        label_names = np.array(list(self.label_names.keys()))
        if index is None:
            index = self.labels
        return label_names[index]

    def __len__(self):
        return len(self.labels)
    
    def update(self, texts, labels=None, labelstr=None, num_feats=None):
        """ Update the data with given texts, labels and optionall
        numeric features.
        TODO: [cleanup] code is replicated in load()
        """
        assert (labels is not None) or (labelstr is not None)
        if self.num_features is not None:
            assert num_feats is not None
            self.num_features = np.vstack((self.num_features, num_feats))
        self.texts.extend(texts)
        if labelstr is None:
            self.labels.extend(labels)
        else:
            for l in labelstr:
                if l not in self.label_names:
                    self.label_names[l] = len(self.label_names)
            self.labels.extend(
                [self.label_names.get(l) for l in labelstr])


    def load(self, path, num_data=None, cat_data=None):
        labels_str = []
        linen = 0
        for l, t in read_csv(path,
                header=(self.class_label, self.text_label),
                sep=self.delimiter):
            linen += 1
            if len(t) < self.minlen or len(t) > self.maxlen:
                warning("Skipping: line {}...".format(linen))
                continue
            labels_str.append(l)
            self.texts.append(t)
            if l not in self.label_names:
                self.label_names[l] = len(self.label_names)
        self.labels.extend(
            [self.label_names.get(l) for l in labels_str])
        if num_data is not None:
            self.num_features = []
            open_file = open
            if num_data.endswith('.gz'):
                open_file = gzip.open
            with open_file(num_data, 'rt') as fp:
                for line in fp:
                    if line.startswith('### '): continue
                    self.num_features.append([float(x)
                        for x in line.strip().split()])
            assert len(self.labels) == len(self.num_features)
            scaler = StandardScaler()
            self.num_features = scaler.fit_transform(
                    np.array(self.num_features))
            debug("Loaded extra features {} from {}.".format(
                        self.num_features.shape, self.num_data))

    def stats(self, histogram=None, most_common=0):
        labels_int = list(self.label_names.values())
        labels_str = list(self.label_names.keys())
        label_dist = Counter(self.labels)
        fmt = '{:30s} {:>6d}' + ''.join(['{:10.2f}{:10.2f}{:>7d}{:>7d}']*2)
        wlen_dist_all = []
        clen_dist_all = []
        for li in labels_int:
            clen_dist = np.array([len(x) for i, x in enumerate(self.texts)\
                if self.labels[i] == li])
            wlen_dist = np.array([len(self.tokenizer(x)) \
                for i, x in enumerate(self.texts) if self.labels[i] == li])
            clen_dist_all.extend(clen_dist)
            wlen_dist_all.extend(wlen_dist)
            print(fmt.format('{}({}):'.format(labels_str[li], li),
                    label_dist[li],
                    clen_dist.mean(), clen_dist.std(), 
                    clen_dist.min(), clen_dist.max(),
                    wlen_dist.mean(), wlen_dist.std(), 
                    wlen_dist.min(), wlen_dist.max()))
        clen_dist_all = np.array(clen_dist_all)
        wlen_dist_all = np.array(wlen_dist_all)
        print(fmt.format('Total:',  sum(label_dist.values()),
                    clen_dist_all.mean(), clen_dist_all.std(),
                    clen_dist_all.min(), clen_dist_all.max(),
                    wlen_dist_all.mean(), wlen_dist_all.std(),
                    wlen_dist_all.min(), wlen_dist_all.max()))
        if most_common:
            tok_counter = [Counter() for _ in labels_int]
            ch_counter = [Counter() for _ in labels_int]
            for i, txt in enumerate(self.texts):
                tok_counter[self.labels[i]].update(self.tokenizer(txt.lower()))
            # only char bigrams - generally useful for detecting odd things
            # at te beginning or end of documents.
                ch_counter[self.labels[i]].update(get_ngrams(list(txt.lower()), 2,2))
            for li in labels_int:
                lname = list(self.label_names.keys())[li]
                print(lname, 'tokens', 
                        [x[0] for x in tok_counter[li].most_common(most_common)])
                print(lname, 'chars', 
                        [x[0] for x in ch_counter[li].most_common(most_common)])
        if histogram:
            import matplotlib.pyplot as plt
            _, plts = plt.subplots(nrows=1,ncols=2)
            plts[0].hist(clen_dist_all, bins=100)
            plts[0].set_title("Char")
            plts[1].hist(wlen_dist_all, bins=100)
            plts[1].set_title("Word")
            if isinstance(histogram, str):
                fig = plt.gcf()
                fig.savefig(histogram)
            else:
                plt.show()

    def copy(self):
        return self.subset(range(len(self)))

    def subset(self, index):
        subset = TextCData()
        for attr in vars(self):
            if not attr.startswith('_'):
                setattr(subset, attr, getattr(self, attr))
        subset.texts = [self.texts[i] for i in index]
        subset.labels = [self.labels[i] for i in index]
        if self.num_features is not None:
            subset.num_features = self.num_features[index]
        return subset
        


def random_iter(param_space, max_iter=1000):
    """ A generator that returns random drwas from a parameter space.

        param_space is a sequence (name, type, range)
        where type is either 'numeric' or 'categorical',
        and range is a triple (start, stop, step) for
        numeric parameters, and another sequence of
        parameter values to explore.

        the function keeps the set of returned parameter values, and
        if an already returned parameter set is drawn max_iter times,
        it terminates.
    """
    seen = set()
    rejected = 0
    while True:
        params = []
        for param, type_, seq in param_space:
            if 'numeric'.startswith(type_):
                try:
                    start, stop, step = seq
                    p_range = np.arange(start, stop + step, step).tolist()
                except:
                    start, stop = seq
                    p_range = np.arange(start, stop + 1, 1).tolist()
                rval = random.choice(p_range)
                params.append((param, rval))
            elif 'categorical'.startswith(type_):
                params.append((param, random.choice(seq)))
        param_hash = md5(str(params).encode()).digest()
        if param_hash not in seen:
            seen.add(param_hash)
            rejected = 0
            yield dict(params)
        else:
            rejected += 1
            if rejected == max_iter:
                info("More than {} iterations with already drawn parameters. "
                      "The search space is probably exhausted".format(max_iter))
                return

def grid_iter(param_space):
    p_vals = []
    for param, type_, seq in param_space:
        if 'numeric'.startswith(type_):
            try:
                start, stop, step = seq
                p_range = np.arange(start, stop + step, step).tolist()
            except:
                start, stop = seq
                p_range = np.arange(start, stop + 1,  1).tolist()
            p_vals.append(p_range)
        elif 'categorical'.startswith(type_):
            p_vals.append(seq)
    for params in itertools.product(*p_vals):
        yield dict(zip((p[0] for p in param_space), params))

def _str_to_param_space(paramstr):
    paramstr = str(paramstr)
    try:
        with open(paramster, 'r') as fp:
            params = eval(fp.read().strip())
    except:
        try:
            params = eval(paramstr)
        except:
            params = '(' + paramstr + ')'
    return params

def read_logs(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            if (len(line) > 1 and line[0] != '#'):
                log_data = json.loads(line)
                tuned_params = log_data['params']
                model_params = log_data['model_params']
                scores = log_data['scores']

                yield tuned_params, scores, model_params

class TextC(object):
    PARAMS = {
            'name': 'textc',
            'baseline': 'random',
            'adapt_thresh': 0.0,
    }
    """ Base text classifier class.
    """
    def __init__(self, arg_parser=True, **params):
        self._set_defaults()
        if arg_parser:
            self.arg_parser = self._setup_arg_parser()
        self._trained = True # Always for baselines
        self.set_params(**params)

    def _set_defaults(self):
        for k,v in self.PARAMS.items():
            setattr(self, k, v)

    def get_params(self):
        return {k:getattr(self, k) for k in self.PARAMS \
                    if not k.startswith('_')}

    def set_params_str(self, s):
        val_dict = dict()
        for pval in s.split(','):
            p, val = pval.split('=')
            try:
                val_dict[p] = eval(val)
            except:
                val_dict[p] = val
        self.set_params(**val_dict)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.PARAMS:
                warning("Ignoring unknown parameter {}.".format(k))
            else:
                old_v = getattr(self, k)
                if v != k:
                    setattr(self, k, v)
                    self._trained = False

    def fit(self, train=None):
        """ Fit the model.

        This should be overridden in the real text classifer classes.
        """
        self._trained = True

    def _predict_k_fold(self, train, k=10, decision_func=False):
        #TODO: pass the parameter k
        splits = StratifiedKFold(n_splits=k, shuffle=True)
        folds = list(splits.split(train.texts, train.labels))
        predictions = [None for _ in range(len(train.labels))]
        if decision_func:
            decision_val = [None for _ in range(len(train.labels))]

        for ti, vi in folds:
            if decision_func:
                pred, dec, _, _ = self.predict(test=train.subset(vi), 
                                train=train.subset(ti), decision_func=True)
            else:
                pred = self.predict(test=train.subset(vi), 
                                train=train.subset(ti))
            for i, j in enumerate(vi):
                predictions[j] = pred[i]
                if decision_func:
                    decision_val[j] = dec[i]
        if decision_func:
            return predictions, decision_val
        return predictions

    def _predict(self, test, train=None, decision_func=False):
        """ Return predictions for the testset. Implements baselines.

        Here we only return the numeric indeices for the labels. To
        obtain symbolic names, use predict().

        This should be overridden in the real text classifer classes.
        """
        if train and not self._trained: 
            self.fit(train)
        label_set = list(train.label_names.values())
        test_len = len(test.texts)
        if self.baseline == 'random':
            predictions = np.random.choice(label_set, size=test_len)
        elif self.baseline == 'majority':
            predictions = test_len * [Counter(train.labels).most_common(1)[0][0]]
        elif self.baseline == 'random_sample':
            prob = Counter(train.labels) 
            prob = np.array([prob[l] for l in label_set]) / sum(prob.values())
            predictions =  np.random.choice(label_set,
                    size=test_len,
                    p=prob)
        if decision_func:
            return predictions, None
        return predictions

    def predict(self, test=None, train=None, label_names=False,
            decision_func=False,
            score=False, conf_mat=False):
        if test is None:
            predictions = self._predict_k_fold(train, decision_func=decision_func)
        else:
            if self.adapt_thresh != 0.0:
                info("Adaptive predictions enabled")
                if train is None: train = self._training_data
                assert train is not None
                pred, dec_val = self._predict(test, train, decision_func=True)
                texts_add, labels_add, num_add = [], [], []
                for i, v in enumerate(dec_val):
                    if len(dec_val.shape) == 1: # binary
                        pick = abs(v) > self.adapt_thresh
                    else: 
                        pick = len(np.argwhere(v > self.adapt_thresh).flatten()) == 1
                    if pick:
                        texts_add.append(test.texts[i])
                        labels_add.append(pred[i])
                        if test.num_features is not None:
                            num_add.append(test.num_features[i]) 
                if not num_add: num_add = None
                train_aug = train.copy()
                train_aug.update(texts_add, labels_add, num_feats=num_add)
                info("Retraining with {} new trainng instances".format(len(texts_add)))
                predictions = self._predict(test, train_aug, decision_func=decision_func)
            else:
                predictions = self._predict(test, train, decision_func=decision_func)

        decision_val = None
        if decision_func:
            predictions, decision_val = predictions

        if label_names and self._training_data.label_names:
            label_names = list(self._training_data.label_names)
            predictions = [label_names[i] for i in predictions]

        sc = None
        if score:
            sc = self._score(test.symbolic_labels(), predictions, 
                    negative_class=train.negative_class)

        cf = None
        if conf_mat:
            from sklearn.metrics import confusion_matrix
            cf = confusion_matrix(test.symbolic_labels(), predictions)
        return predictions, decision_val, sc, cf

    def _score(self, gold, pred,
            scores={'precision', 'recall', 'f1-score'},
            negative_class=None,
            average=None):
        """ Return the score for the testset.
        """
        from sklearn.metrics import precision_recall_fscore_support as prfs

        if average is None:
            average = 'macro'
            if negative_class:
                average = 'binary'

        scores = [sc \
                    if ':' in sc or \
                        sc not in {'precision', 'recall', 'f1-score'}\
                    else ':'.join((sc, average))\
                    for sc in scores]
        scores = {k:None for k in scores}
        for sc_avg in list(scores):
            if ':' in sc_avg:
                sc, avg = sc_avg.split(':')
            else:
                sc = sc_avg
                avg = None
            if scores[sc_avg] is not None:
                continue
            if sc not in {'precision', 'recall', 'f1-score', 'accuracy'}:
                warning("Skipping unknown score `{}'.".format(sc))
                continue
            if sc in {'precision', 'recall', 'f1-score'}:
                if avg not in {'binary', 'micro', 'macro'}:
                    warning("Skipping `{}': unknown avgeraging method."
                            .format(sc_avg))
                    continue
                p, r, f, _ = prfs(gold, pred, average=avg)
                scores[':'.join(('precision', avg))] = p
                scores[':'.join(('recall', avg))] = r
                scores[':'.join(('f1-score', avg))] = f
            if sc == 'accuracy':
                from sklearn.metrics import accuracy_score
                scores['accuracy'] = accuracy_score(gold, pred)
        return {k:v for k,v in scores.items()}

    def score(self,  test=None, train=None,
            scores={'precision', 'recall', 'f1-score'},
            average=None):
        """ Return the score for the testset.
        """
        from sklearn.metrics import precision_recall_fscore_support as prfs
        pred, _, _, _ = self.predict(test, train)
        if test:
            gold = test.labels
        else:
            gold = train.labels

        return self._score(gold, pred, scores=scores,
                negative_class=train.negative_class, 
                average=average)

    def ttt(self, method):
        if method == 'grid':
            param_iter = grid_iter(params)


    def tune(self, param_space,
            train, dev=None,
            method=None,
            round_digits=None,
            max_iter=-1,
            optimize=None,
            scores=None,
            k=None, split_r=0.2, n_splits=1,
            save=sys.stdout, skip_params=None):
        """ Fit and evaluate a model repeatedly,
        the best one based on params.
        
        Args:
            param_space: a dict-like object whose keys are the
                    parameter names and values are tuples of
                    (type, range) or (type, list). 'type' is one of
                    'numeric', 'categorical'. For numeric
                    parameters the range is defined as [begin, end]
                    or [begin, end, step] is required. The former
                    is interpreted as range of integers in the
                    range [begin, end]. For 'categorical'
                    parameters, a list of values is required.
            train:  TextCData object used for training. If no test
                    set is given, it is used both for training
                    and testing
            dev:    same as 'train', used as development set
            method: search method: 'grid', 'random' or a iterable
                    that returns values from the option range.
            max_iter: maximum number of fit/predict/eval iterations
            k:      Use k-fold CV.
            split_r: ratio of the held-out set, ignored if test set or
                     argument `k' is given
            n_splits: number of splits, sklearn StratifiedShuffleSplit is
                    used for n-splits
            save:   file-like object save the results after each iteration
            skip_params: A sequence of dictionaries containing individual
                    parameter values to skip. Useful for resuming a search. 
            optimize: optimize using given metric. 
            scores: the scores to report/log.
        """
        if method is None: method = 'grid'
        if optimize is None: optimize = 'f1-score:macro'
        if scores is None: scores = ['precision', 'recall', 'f1-score']

        param_space = [p for p in _str_to_param_space(param_space) \
                            if p[0] in self.get_params()]
#        if not param_space:
#            info('Tunable parameter list is empty')
#            return
        if method == 'grid':
            param_iter = grid_iter(param_space)
        elif method == 'random':
            param_iter = random_iter(param_space)
        else:
            param_iter = method(params)

        if dev is not None:
            tune_str = "development data".format()
        else:
            if k:
                splits = StratifiedKFold(n_splits=k, shuffle=True)
                tune_str = "{}-fold CV".format(k)
            else:
                splits = StratifiedShuffleSplit(
                                    n_splits=n_splits, test_size=split_r)
                tune_str = "{} splits of {} ratio".format(n_splits, split_r)
            trn_splits, val_splits = [], []
            for ti, vi in splits.split(train.texts, train.labels):
                trn_splits.append(train.subset(ti))
                val_splits.append(train.subset(vi))

        def param_to_str(p_dict):
            p_list = sorted(tuple(p_dict.items()))
            p_fmt = "{}={} " * len(p_list)
            return p_fmt.format(*[x for t in p_list for x in t])

        if skip_params:
            skip_params = set([md5(param_to_str(p).encode()).digest() \
                                    for p in skip_params])
        else:
            skip_params = set()


        best_mean = 0.0
        best_sc = None
        best_param = None
        for param in param_iter:
            scores = []
            p_str = param_to_str(param)
            p_hash = md5(p_str.encode()).digest()
            if p_hash in skip_params:
                info('Skipping: {}'.format(p_str))
                continue
            info('Tuning with {}: {}'.format(tune_str, p_str))
            self.set_params(**param)
            if dev is not None:
                sc = self.score(dev, train=train)
                scores.append(sc)
            else:
                for i in range(len(trn_splits)):
                    sc = self.score(val_splits[i], train=trn_splits[i])
                    scores.append(sc)
            sc_names = scores[0].keys()
            scores = {k:[sc[k] for sc in scores] for k in sc_names}
            sc_mean = np.array(scores[optimize]).mean()
            if sc_mean > best_mean:
                best_mean = sc_mean
                best_sc = scores
                best_param = param
            if save:
                json.dump({'params': param,
                           'model_params': self.get_params(),
                           'scores': scores},
                        save, ensure_ascii=False)
                print('', file=save, flush=True)
            max_iter -= 1
            if max_iter == 0:
                break
            stop_file = '.stop-tune' + str(os.getpid())
            if os.path.isfile(stop_file):
                os.remove(stop_file)
                break
        return best_param, best_sc

    def _setup_arg_parser(self):
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument('--input', '-i', help="Path to the training data")
        ap.add_argument('--test', '-t', help="Path to the testing data")
        ap.add_argument('--unlabeled-input', '-u',
            help="Path to unlabeled training data")
        ap.add_argument('--unlabeled-num-input', '-U',
            help="Path to numeric features for the unlabeled data")
        ap.add_argument('--input-numeric', '-N', 
                help="Path to (optional) additional numeric features")
        ap.add_argument('--test-numeric', '-M', 
                help="Path to (optional) additional numeric features")
        ap.add_argument('--class-label', '-L', default='label',
                help="Label of the column corrsponding to the class.")
        ap.add_argument('--text-label', '-T', default='text',
                help="Label of the column corrsponding to the text.")
        ap.add_argument('--delimiter', '-D', default='\t',
                help="Delimiter used in input files")
        ap.add_argument('--output', '-o', default='-',
                            help="Output file. `-' means stdout.")
        ap.add_argument('--negative-class', 
                            help="The negative class label.")

        ap.set_defaults(command='tune')
        subp = ap.add_subparsers(help="Command")

        tunep = subp.add_parser('tune')
        tunep.set_defaults(command='tune')
        tunep.add_argument('params',
                help=('A string or a filename defining parameter space '
                      'to be searched.'
                      'String must be interpretable by python eval() '
                      'as a sequence whose members are triples of '
                      '(name, type, values). '
                      'If "type" is "numeric", values should specify a range '
                      '(start, stop, step), otherwise ("categorical"), '
                      'a sequence of values. '
                      'Example: (("C", "real", (0.5, 1.5, 0.1)), '
                      '("lowercase", "cat", ("word", "char", "both"))). '
                      'If "params" is a readable file, the string is read '
                      'from the file.'
                      ))
        tunep.add_argument('--search-method', '-s', choices=('grid', 'random'),
                default='grid', help="Method used for hyper parmeter search")
        tunep.add_argument('--optimize',
                help=('A string of the form "score" or "score:averaging" '
                      'Currently supported scores are '
                      'accuracy, precision, recall, f1-score, '
                      'and supproted averaging methods are '
                      'micro, macro and binary. '
                      'If averaging is not specified it is set to '
                      'macro or binary depending on the classification task.'
                    ))
        tunep.add_argument('--max-iter', '-m', type=int, default=-1,
                help=('Maximum number of hyperparameter combinations '
                      'to compare. Default (-1) means until '
                      'the search space is exhausted'))
        tunep.add_argument('--k-folds', '-k', type=int, default=None,
                help=('Use k-fold cross validation. '
                      'Ignored if -t is given.'))
        tunep.add_argument('--test-ratio', '-r', type=float, default=0.2,
                help=('Ratio of held-out data. '
                      'Ignored if --test option is given'))
        tunep.add_argument('--n-splits', '-n', type=int, default=1,
                help=('Number of splits. Ignored if -t or -k is given'))
        tunep.add_argument('--save', '-S', metavar='LOGFILE', 
                help=('Save intermediate parameter values and scores '
                      'to given log file. '
                      'Use - for standard output'))
        tunep.add_argument('--resume-from', '-R', metavar='LOGFILE',
                help=('Resume tuning, skipping the parameters that '
                      'are logged in the log file.'))

        predp = subp.add_parser('predict')
        predp.set_defaults(command='predict')
        predp.add_argument('params',
                help=('A string that can be interpreted by python eval() '
                      'as a sequence whose members are pairs '
                      'of, parameter=value'))
        predp.add_argument('--score', action='store_true',
                            help='Also print out the scores.')
        predp.add_argument('--only-score', action='store_true',
                            help='Print out only the scores.')
        predp.add_argument('--conf-matrix', action='store_true',
                            help='Also print out the confusion matrix.')
        predp.add_argument('--output-decision-value', action='store_true',
                            help=('Also return decision function values'))

        testp = subp.add_parser('score')
        testp.set_defaults(command='score')
        testp.add_argument('params',
                help=('A string that can be interpreted by python eval() '
                      'as a sequence whose members are pairs '
                      'of, parameter=value'))

        #TODO: this should not be here
        statsp = subp.add_parser('stats')
        statsp.set_defaults(command='stats')
        statsp.add_argument('--histogram', '-H',
                    const=True, metavar='FILE', nargs='?',
                    help="Plot a histogram, optionally to FILE.")
        statsp.add_argument('--most-common', type=int, default=0,
                            help="Also print most-common tokens")

        return ap

if __name__ == "__main__":

    m = TextC()
    opt = m.arg_parser.parse_args()
    
    trn = TextCData(opt.input, num_data=opt.input_numeric,
            class_label=opt.class_label,
            text_label=opt.text_label, sep=opt.delimiter)
    tst = None
    if opt.test:
        tst = TextCData(opt.test, num_data=opt.test_numeric,
                labels=trn.label_names,
                class_label=opt.class_label,
                text_label=opt.text_label, sep=opt.delimiter)

    if opt.command == 'stats':
        trn.stats(histogram=opt.histogram, most_common=opt.most_common)
    elif opt.command == 'score':
        print(m.score(tst, train=trn,
                scores=['accuracy', 'precision:micro', 'precision:macro']))
    elif opt.command == 'predict':
        pred = m.predict(tst, train=trn)
        if opt.output == '-':
            fp = sys.stdout
        else:
            fp = open(opt.output, 'w')
        for p in pred:
            print(p, file=fp)
        if fp != sys.stdout:
            fp.close()
    elif opt.command == 'tune':
        skip = None
        if opt.resume_from:
            skip = [x for (x, _, _) in read_logs(opt.resume_from)]

        savefp = None
        if opt.save:
            if opt.save == '-':
                savefp = sys.stdout
            else:
                if skip and opt.resume_from == opt.save:
                    savefp = open(opt.save, 'a')
                else:
                    savefp = open(opt.save, 'w')
        best_param, best_sc = m.tune(opt.params, 
                      trn, dev=tst,
                      method=opt.search_method,
                      max_iter=opt.max_iter,
                      k=opt.k_folds,
                      split_r=opt.test_ratio,
                      n_splits=opt.n_splits,
                      save=savefp,
                      optimize=opt.optimize,
                      skip_params=skip)
        print('best params:', best_param)
        print('best score:', best_sc)
        if savefp and savefp != sys.stdout:
            savefp.close()
