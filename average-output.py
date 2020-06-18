#!/usr/bin/env python3

import sys
import numpy as np
from collections import Counter

labels = list()
for f in sys.argv[1:]:
    with open(f, 'rt') as fp:
        a = None
        d = list()
        lab = list()
        for line in fp:
            if 'Label names:' in line: continue
            l, n = line.strip().split('\t', 1)
            lab.append(l)
            d.append([float(x) for x in n.split()])
    labels.append(lab)
    if a is None:
        a = np.array(d)
    else:
        a += np.array(d)
labels = [''.join(x) for x in zip(*labels)]
a /= len(sys.argv[1:])

single_label = list()
for l in labels:
    counts = Counter(l).most_common(2)
    if len(counts) == 1:
        single_label.append(counts[0][0])
    else:
        (first, n), (second, m) = counts
        if n == m:
            single_label.append(l[0])
        else:
            single_label.append(first)

fmt= "{}\t{}" + a.shape[1] * "\t{}"
for i, l in enumerate(labels):
    print(fmt.format(l, single_label[i], *a[i,:].tolist()))
