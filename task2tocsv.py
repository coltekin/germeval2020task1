#!/usr/bin/env python3

import csv
import sys

dataset = sys.argv[1]

texts = "{0}_data_sets/psychopred_task2_{0}_texts.tsv".format(dataset)
labels = "{0}_data_sets/psychopred_task2_{0}_labels.tsv".format(dataset)

d = dict()
with open(texts, 'rt') as f:
    csvr = csv.reader(f, delimiter='\t', escapechar='\\')
    next(csvr)
    for uuid, text in csvr:
        d[uuid] = {'text': text}

if dataset == 'test':
    with open(sys.argv[2], 'wt') as f:
        csvw = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow("uuid text".split())
        for uuid, data in d.items():
            csvw.writerow([uuid, data['text']])
else:
    with open(labels, 'rt') as f:
        csvr = csv.reader(f, delimiter='\t', escapechar='\\')
        next(csvr)
        for uuid, motive, level in csvr:
            d[uuid].update({'motive': motive, 'level': level})

    with open(sys.argv[2], 'wt') as f:
        csvw = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow("uuid label motive level text".split())
        for uuid, data in d.items():
            lab = "-".join((data['motive'], data['level']))
            csvw.writerow([uuid, lab, data['motive'], data['level'], data['text']])
