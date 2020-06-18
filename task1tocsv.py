#!/usr/bin/env python3

import csv
import sys

dataset = sys.argv[1]

texts = "{0}_data_sets/psychopred_task1_{0}_answers.tsv".format(dataset)

#student_ID      image_no        answer_no       UUID    MIX_text
d = dict()
with open(texts, 'rt') as f:
    csvr = csv.reader(f, delimiter='\t', escapechar='\\')
    next(csvr)
    for stuid, img, ans, uuid, text in csvr:
        d[uuid] = {'stuid': stuid, 'img': img, 'answer': ans, 'text': text}


with open(sys.argv[2], 'wt') as f:
    csvw = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    csvw.writerow("stuid img answer text motive level label".split())
    stu_texts = dict()
    for uuid, data in d.items():
        stuid = data['stuid']
        csvw.writerow([stuid, data['img'], data['answer'], data['text'], "_", "_", "_"])
        if stuid in stu_texts:
            stu_texts[stuid] += '\n' + data['text']
        else:
            stu_texts[stuid] = data['text']

with open(sys.argv[2] + "-perstudent", 'wt') as f:
    csvw =  csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    csvw.writerow("stuid text motive level label".split())
    for stuid, text in stu_texts.items():
        csvw.writerow([stuid, text, "_", "_", "_"])
