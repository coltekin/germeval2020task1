# Linear systems for predicting academic achievement, and motivational style

This repository contains the code used for participating in
[GermEval 2020 task 1](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2020-cognitive-motive.html)
on the classification and regression of cognitive and Motivational
style from text.

The task consisted of two subtasks.
The first task is a regression task for predicting a ranking
based on high school grades and IQ scores.
The second task is about automatically annotating  
the texts obtained during a psychometric tests
(Operant Motive Test, OMT).
Please see the
[shared task web page](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2020-cognitive-motive.html)
for detailed description, and the data.
The data as provided by the shared task organizers
should be placed in the 'data/' directory
(or otherwise change the default paths in the scripts).

## Reproduce the results in the paper

The reproduction is not completely automated.
There are a few manual steps involved.
However, since both systems are based on simple linear models,
the variation in the results are rather low.

In case you want to use the code but having difficulties, 
please do not hesitate to contact the author.

### Subtask 1
For subtask 1, the main script is `textr.py`.
The following command is used to tune the individual regression models
for each target score (`german`, `english`, `math`, `lang`, `logic`
of `sum` for predicting the sum of them directly)
in the data.

```
python3 textr.py -AZ -M 4000 tune <target> -S logs/regression \
    -p "('w_ngmax', 'n', (2,4,1)), ('c_ngmax', 'n', (2,8,1)), ('alpha', 'n', (0.001, 50.0, 0.1)), ('lowercase', 'c', ('word', 'none'))"
```
This command performs 4000 random draws from the specified parameter
ranges, trains the model using 5-fold CV with drawn parameter,
and writes the scores obtained along with parameter values to
`logs/regression-<target>.log`, where `<target>` is one of the scores
above.

The script `read-logs-r.py` reads the logs and prints a sorted list of
scores and the output (see `-h` for command line usage).

Once tuning is complete, the command
```
python3 ./textr.py -o predictions.txt -AZ -T 10 -S logs/regression predict <target>
```
uses average of the top 10 best models in the log file,
trains the model on the training data predicting the results
for the development data
(use `-t` for training on train+dev sets and testing on the test data),
prints out the performance scores,
and writes the predicted ranks to `predictions.txt`.

### Subtask 2

The main script for subtask 2 is `bong.py`.
This script accepts tab-separated input files.
So data needs to be converted first.
The scripts `task1tocsv.py`, and `task2tocsv.py` do the conversion.

The rest of the procedure is similar to task 1 script
(see `./bong.py -h` for documentation),
except, `predict` with  this script predicts using only the
hyperparameter settings provided on the command line.
To get output averaged over multiple hyperparameter settings,
you need to use `average-output.py`.

