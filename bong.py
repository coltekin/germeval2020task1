#!/usr/bin/env python3
"""Train/tune 'bag-of-n-grams' (BoNG) models.
   These are essentially wrappers areound sklearn linear classifiers.
"""

import sys, re
import numpy as np
import logging
from scipy.sparse import hstack
from logging import debug, info, warning, basicConfig
basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

from textc import TextCData, TextC, get_ngrams, _str_to_param_space

class BongVectorizer:
    W_TOK = re.compile("\w+|[^ \t\n\r\f\v\w]+").findall
    PARAMS = {'min_df': 1,
              'c_ngmin': 1, 'c_ngmax': 1,
              'w_ngmin': 1, 'w_ngmax': 1,
              'lowercase': None,
              'vectorizer': 'tfidf',
              'solver': 'liblinear',
              'b': 0.75,
              'k1': 2.0}

    def __init__(self, cache_vectors=0, **kwargs):
        self.cache_file = kwargs.get('cache_file', None)
        self.v = None
        self.tokenizer = self.W_TOK
        self._trianed = False
        self._training_data = None
        for k, v in self.PARAMS.items(): setattr(self, k, v)
        self.cache_vectors = cache_vectors
        if cache_vectors:
            self._vector_cache = dict()
            self._vector_counter = 0
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.PARAMS:
                warning("Ignoring unknown parameter {}.".format(k))
            else:
                old_v = getattr(self, k)
                if v != old_v:
                    setattr(self, k, v)

    def get_params(self, **kwargs):
        return {k: getattr(self, k) for k in self.PARAMS}

    def _init_vectorizer(self):
        if self.vectorizer not in {'tfidf', 'bm25'}:
            info("Unknown/unimplemented vectorizer {}"
                 ", falling back to TF-IDF".format(self.vectorizer))
            self.vectorizer = 'tfidf'
        if self.vectorizer == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.v = TfidfVectorizer(sublinear_tf=True,
                                     min_df=self.min_df,
                                     analyzer=self.doc_analyzer)
        elif self.vectorizer == 'bm25':
            from bm25 import BM25Vectorizer
            self.v = BM25Vectorizer(sublinear_tf=True,
                                     min_df=self.min_df,
                                     analyzer=self.doc_analyzer,
                                     b = self.b, k1 = self.k1)

    def fit(self, data):
        self._init_vectorizer()
        self.v.fit(data.texts)
        self._training_data = data
        self._trained = True

    def _check_cache(self, data, trn_data=None):
        if trn_data is None: trn_data = self._training_data
        if not trn_data or not self.cache_vectors:
            return None
        paramstr = ",".join("=".join((k,str(v))) for k,v in
                sorted(self.PARAMS.items()))
        self._vector_counter += 1
        cached = self._vector_cache.get((trn_data._id, data._id, paramstr),None)
        if cached:
            cached[0] = self._vector_counter
            return cached[1]
        else:
            return None

    def _update_cache(self, data, v):
        if not self.cache_vectors:
            return None
        paramstr = ",".join("=".join((k,str(v))) for k,v in
                sorted(self.PARAMS.items()))
        self._vector_cache[(self._training_data._id, data._id, paramstr)] = \
                [self._vector_counter, v]
        if len(self._vector_cache) > self.cache_vectors:
            oldest = sorted(self._vector_cache,
                    key=lambda x: self._vector_cache[x][0])[0]
            del self._vector_cache[oldest]

    def fit_transform(self, data):
        cached = self._check_cache(data, data)
        if cached is not None:
            info("Using cached vectors (f/t)")
            self._training_data = data
            self._trained = False
            return cached

        self._init_vectorizer()
        self._training_data = data
        v = self.v.fit_transform(data.texts)
        self._trained = True
        self._update_cache(data, v)
        return v

    def transform(self, data):
        cached = self._check_cache(data)
        if cached is not None:
            info("Using cached vectors (t)")
            return cached
        else:
            if not self._trained:
                self.fit_transform(self._training_data)
            v = self.v.transform(data.texts)
            self._update_cache(data, v)
            return v

    def doc_analyzer(self, doc):
        """ Convert document to word/char ngrams with optional
            case normaliztion.
        """

        if self.lowercase is None or self.lowercase == 'none':
            lowercase = set()
        elif self.lowercase in {'both', 'all'}:
            lowercase = {'char', 'word'}
        else: lowercase = {self.lowercase}

        # character n-grams
        if 'char' in lowercase:
            docfeat = get_ngrams(list(doc.lower()),
                    self.c_ngmin, self.c_ngmax)
        else:
            docfeat = get_ngrams(list(doc),
                    self.c_ngmin, self.c_ngmax)
        # word n-grams
        if 'word' in lowercase:
            docfeat.extend(get_ngrams(self.tokenizer(doc.lower()),
                                    self.w_ngmin, self.w_ngmax,
                                    suffix="⅏", separator=" "))
        else:
            docfeat.extend(get_ngrams(self.tokenizer(doc),
                                    self.w_ngmin, self.w_ngmax,
                                    suffix="⅏", separator=" "))
        return docfeat


class BongC(TextC):
    PARAMS = {
            'classifier': 'svm',
            'vectorizer': 'tfidf',
            'multi_class': 'ovr',
            'C': 1.0,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'num_mix': 1.0,
            'max_iter': 20000,
            'random_state': None,
            'dual': True,       # for SVMs
            'n_estimators': 50, # for RF
            'adapt_thresh': 0.0,
    }
    def __init__(self, **kwargs):
        self.v = BongVectorizer()
        for k,v in self.v.get_params().items():
            self.PARAMS[k] = v
        super().__init__(self)
        self._trained = False
        self.model = None
        self._training_data = None
        #
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.PARAMS:
                warning("Ignoring unknown parameter {}.".format(k))
            else:
                if k in self.v.get_params():
                    self.v.set_params(**{k: v})
                old_v = getattr(self, k)
                if v != old_v:
                    debug('Setting {}, old = {}, new = {}'.format(k, old_v,v ))
                    setattr(self, k, v)
                    self._trained = False

    def fit(self, train, val=None):
        info("Converting documents to BoNG vectors")
        docs = self.v.fit_transform(train)
        info("Number of features: {}".format(
                            len(self.v.v.vocabulary_)))

        if self._training_data == train and self._trained:
            info("Skipping training the model, parameters and data did not change.")
            return
        if self.classifier == 'lr':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression
            clf_params = {'C', 'multi_class', 'dual', 'class_weight',
                    'random_state', 'max_iter', 'solver'}
        elif self.classifier == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier
            clf_params = {'class_weight', 'n_estimators', 'n_jobs', 'random_state'}
        else:
            from sklearn.svm import LinearSVC
            clf = LinearSVC
            clf_params = {'C', 'multi_class', 'dual', 'class_weight',
                    'random_state', 'max_iter'}

        clf_params = {k:v for k,v in self.get_params().items()\
                        if k in clf_params}
        self.model = clf(**clf_params)

        if self.multi_class:
            if self.multi_class == 'ovo':
                from sklearn.multiclass import OneVsOneClassifier
                self.model = OneVsOneClassifier(self.model, n_jobs=self.n_jobs)
            elif self.multi_class == 'ovr':
                from sklearn.multiclass import OneVsRestClassifier
                self.model = OneVsRestClassifier(self.model, n_jobs=self.n_jobs)

        if train.num_features is not None:
            docs = hstack((docs, self.num_mix * train.num_features), format="csr")
        info("Fitting the model {}".format(docs.shape))
        self.model.fit(docs, np.array(train.labels))
        self._training_data = train
        self._trained = True

    def _predict(self, test, train=None, decision_func=False):
        if train: self.fit(train)
        x = self.v.transform(test)
        if self._training_data.num_features is not None\
                and test.num_features is not None:
            x = hstack((x, self.num_mix * test.num_features), format="csr")
        predictions = self.model.predict(x)
        if decision_func:
            decision_val = self.model.decision_function(x)
            return predictions, decision_val
        return predictions

if __name__ == '__main__':
    from textc import read_logs
    m = BongC()
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
    trn_u = None
    if opt.unlabeled_input:
        trn_u = TextCData(opt.unlabeled_input,
                    num_data=opt.unlabeled_numeric,
                    labels=trn.label_names,
                    class_label=opt.class_label,
                    text_label=opt.text_label, sep=opt.delimiter)

    if opt.command == 'stats':
        trn.stats(histogram=opt.histogram)
    elif opt.command == 'score':
        m.set_params_str(opt.params)
        print(m.score(tst, train=trn,
                scores=['accuracy', 'precision:micro', 'precision:macro']))
    elif opt.command == 'predict':
        m.set_params_str(opt.params)
        pred, dec_val, score, conf_mat = m.predict(tst, train=trn,
                label_names=True, decision_func=opt.output_decision_value,
                score=(opt.score or opt.only_score),
                conf_mat=opt.conf_matrix)
        debug("only score: {}".format(opt.only_score))
        if not opt.only_score:
            if opt.output == '-':
                fp = sys.stdout
            else:
                fp = open(opt.output, 'w')
            if dec_val is not None:
                dec_val = np.array(dec_val)
                for i, p in enumerate(pred):
                    if len(dec_val.shape) == 1:
                        dec_str = str(dec_val[i])
                    else:
                        dec_str = " ".join([str(x) for x in dec_val[i]])
                    print("{}\t{}".format(p, dec_str), file=fp)
                print('Label names:', trn.label_names, file=fp)
            else:
                for p in pred:
                    print(p, file=fp)
            if fp != sys.stdout:
                fp.close()
        if score is not None:
            print(score, file=sys.stderr)
        if conf_mat is not None:
            labels = list(trn.label_names.keys())
            max_lab = max([len(x) for x in labels])
            max_cell = int(np.log10(conf_mat.max()))+1
            fmt = "{:>" + str(max_lab) + "}" 
            fmt += (" {:>" + str(max(max_lab, max_cell)) + "}") * len(labels)
            print(fmt.format(" ", *labels))
            for i, row in enumerate(conf_mat):
                print(fmt.format(labels[i], *row))
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
