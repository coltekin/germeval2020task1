#!/usr/bin/env python3
import csv
import os
import re
import logging
import numpy as np
from collections import Counter
from logging import debug, info, warning, basicConfig
basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
from scipy.stats import describe, zscore

score_names = "german english math lang logic".split()
class Task1data:
    def __init__(self, path='../data/'):
        self.path = path
        self.data = dict()
        self.questions = tuple()
        self.extra_data = False

    def has_outcome(self):
        for sid in self.data: break
        return 'sum' in self.data[sid]

    def load(self, data_set='train', normalize=False):
        dir_ = os.path.join(self.path, '{}_data_sets'.format(data_set))
        answers = os.path.join(dir_, 
                    "psychopred_task1_{}_answers.tsv".format(data_set))
        ranks = os.path.join(dir_, 
                    "psychopred_task1_{}_student_ranks.tsv".format(data_set))
        students = os.path.join(dir_, 
                    "psychopred_task1_{}_students.tsv".format(data_set))
        extra = os.path.join(dir_, 
                    "psychopred_task1_{}_xtra.tsv".format(data_set))

        d, qset = dict(), set(self.questions)
        with open(answers, 'rt') as fp:
            _ = next(fp).strip().split('\t')
            for line in fp:
                stid, img, ans, uuid, text = line.strip().split('\t')
                qid = '_'.join((img,ans))
                if stid not in d: d[stid] = {'text': dict(), 'extra': dict()}
                d[stid]['text'][qid] = text
                qset.add(qid)
        self.questions = tuple(sorted(qset))

        if os.path.exists(extra):
            self.extra_data = True
            with open(extra, 'rt') as fp:
                _ = next(fp)
                for line in fp:
                    row = line.strip().split('\t')
                    stid = row[0]
                    d[stid]['extra'] = np.array([float(x) for x in row[1:]])


        if os.path.exists(students):
            with open(students, 'rt') as fp:
                _ = next(fp)
                for line in fp:
                    row = line.strip().split('\t')
                    stuid = row[0]
                    row = [float(x) for i,x in enumerate(row) if i != 0]
                    d[stuid].update({k:row[i]\
                            for i,k in enumerate(score_names)})
                    d[stuid]['sum'] = sum(row)
        self.data.update(d)
        if os.path.exists(students):
            if normalize:
                stuids = self.data.keys()
                scores = [[self.data[stu][n] for n in score_names] \
                                for stu in stuids]
                scores = zscore(np.array(scores), axis=0) 
                for i, stuid in enumerate(stuids):
                    for j, sc in enumerate(score_names):
                        self.data[stuid][sc] = scores[i,j]
                    self.data[stuid]['sum'] = scores[i,:].sum()
                if self.extra_data:
                    extrafeat = np.array([self.data[stid]['extra'] \
                                    for stid in stuids])
                    extrafeat = zscore(np.array(extrafeat), axis=0)
                    for i, stid in enumerate(stuids):
                        self.data[stid]['extra'] = extrafeat[i,:].ravel()
            self.update_ranks()

    def update_ranks(self, normalize=False):
        stu_ids = self.data.keys()
        ranks = np.array(range(len(stu_ids)))
        if normalize:
            ranks = zscore(ranks)
        for i, stu in enumerate(
                sorted(stu_ids, key=lambda x: self.data[x]['sum'],
                    reverse=True)):
            self.data[stu]['rank'] = ranks[i]

    def get_data(self, questions=None, return_extra=False, shuffle=False):
        stu_ids = self.data.keys()
        if shuffle:
            stu_ids = np.random.choice(
                            stu_ides, size=len(stu_ids), replace=False)
        if not questions: questions = self.questions
        for stu in stu_ids:
            for q in questions:
                text = self.data[stu]['text'][q]
                scores = None
                if self.has_outcome():
                    scores = {k:self.data[stu][k] for k in score_names}
                    scores['rank'] = self.data[stu].get('rank', None)
                    scores['sum'] = self.data[stu]['sum']
                if return_extra:
                    extra = self.data[stu].get('extra', None)
                    yield stu, text, extra, scores, q
                else:
                    yield stu, text, scores, q

    def get_data2(self, outcome, questions=None, aggregate=False,
            shuffle=False, sep='\n', return_extra=False):
        if not questions: questions = self.questions
        text, out, extrafeat, stuids = list(), list(), list(), list()
        for sid, t, e, sc, q in self.get_data(
                questions=questions, return_extra=True):
            stuids.append(sid)
            extrafeat.append(e)
            text.append(t)
            if sc:
                out.append([sc[x] for x in outcome])
        if aggregate:
            stuid_i = {k:i for i,k in enumerate(self.data.keys())}
            text_combined = ['' for _ in stuid_i]
            out_combined = [None for _ in stuid_i]
            extra_combined = [None for _ in stuid_i]
            for i in range(len(text)):
                si = stuid_i[stuids[i]]
                text_combined[si] += text[i]
                if self.has_outcome():
                    out_combined[si] = out[i]
                else:
                    out_combined = None
                if return_extra:
                    extra_combined[si] = extrafeat[i]
            text, out = text_combined, out_combined
            if return_extra:
                extrafeat = extra_combined
            stuids = sorted(stuid_i, key=lambda x: stuid_i[x])
        text = np.array(text, dtype=object)
        if self.has_outcome():
            out = np.array(out)
        else:
            out = None
        stuids = np.array(stuids, dtype=object)
        if return_extra:
            return (text, np.array(extrafeat)), out, stuids
        else:
            return text, out, stuids


tokenize = re.compile(r"\w+|[^ \t\n\r\f\v\w]+").findall
class TextEncoder:
    def __init__(self, tokenizer=tokenize):
        self.tokenizer = tokenizer
        self.token_map = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.char_map = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.token_length = Counter()
        self.char_length = Counter()
        self.token_count = Counter()
        self.char_count = Counter()
        self.max_tokens, self.max_chars = 0, 0

    def fit(self, data):
        for stu, text, scores, q in data.get_data():
            tokens = self.tokenizer(text)
            self.token_count.update(tokens)
            self.token_length.update([len(tokens)])
            self.char_count.update(text)
            self.char_length.update([len(text)])
            for tok in tokens:
                self.token_map.setdefault(tok, len(self.token_map))
            for ch in text:
                self.char_map.setdefault(tok, len(self.char_map))
            if len(tokens) > self.max_tokens:
                self.max_tokens = len(tokens)
            if len(text) > self.max_chars:
                self.max_chars = len(text)

    def get_seq_len(self, coverage=1.0, tokens=False):
        if coverage > 1.0:
            coverage /= 100
        dist = self.char_length
        if tokens:
            dist = self.token_length
        n = sum(dist.values())
        skip = 0
        prev_len = None
        for length in sorted(dist, reverse=True):
            skip += dist[length]
            if (1.0 - skip / n) <= coverage:
                if prev_len: return prev_len
                else: return length
            prev_len = length

    def vocabulary_size(self):
        return len(self.token_map)

    def labelset_size(self, squash=False):
        if squash:
            return len(self.labelset)
        else:
            return len(self.label_map)

    def transform(self, seq,
            mark_begin=True, mark_end=True, pad=0, pad_at='begin',
            tokenize=False):
        if tokenize:
            seq = self.tokenizer(seq)
        int_seq = [self.token_map.get(tok, self.token_map['<unk>']) \
                for tok in seq]
        if mark_begin:
            int_seq.insert(0, self.token_map['<s>'])
        if mark_end:
            int_seq.append(self.token_map['</s>'])
        if pad:
            n = len(int_seq)
            if n < pad:
                padding =  [self.token_map['<pad>']] * (pad - n)
                if pad_at == 'end':
                    int_seq = int_seq + padding
                elif pad_at == 'begin':
                    int_seq = padding + int_seq 
            if n > pad:
                if pad_at == 'end':
                    int_seq = int_seq[:pad]
                    int_seq[-1] = self.token_map['</s>']
                elif pad_at == 'begin':
                    int_seq = int_seq[-pad:]
                    int_seq[0] = self.token_map['<s>']
        return int_seq





if __name__ == "__main__":
    trn = Task1data()
    trn.load('train')
#    d.load('train', normalize=True)
    dev = Task1data()
    dev.load('dev')

#    d.load('dev')

#    for stu in sorted(d.data, key=lambda x: d.data[x]['rank']):
#        print(d.data[stu]['sum'], d.data[stu]['rank'] )
#    for stu, t, s, q in d.get_data(questions=['15_2']):
#        print(stu, t, s, q)
#        print(stu, s['sum'], s['rank'])

    for d in trn, dev:
        scores = []
        for stu, t, s, q in d.get_data():
            scores.append(s)

        sc_names = score_names + ['sum', 'rank']
        scores = [[sc[n]  for n in sc_names] for sc in scores]
        scores = np.array(scores)
        print(sc_names)
        fmt = ("{}: n={}, min={:.4f}, max={:.4f}, mean={:.4f}, "
               "var={:.10f}, skew={:.4f}, kurtosis={:.4f}")
        for i, sc in enumerate(sc_names):
            dsc = describe(scores[:,i].ravel())
            print(fmt.format(sc, dsc[0], *dsc[1], *dsc[2:]))
