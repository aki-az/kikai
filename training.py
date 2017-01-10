# -*- coding: utf-8 -*-
# for python3
#
# いろいろ学習する
# python training.py {m} {pcsv} {ncsv} {pdump}
# m    : 'p'=パーセプトロン, 'a'=AdaBoost+パーセプトロン, 's'=SVM
# pcsv : 正例のCSVファイル名
# ncsv : 負例のCSVファイル名
# pdump : 学習後のパラメータ
#
# python chk_training.py {m} {pcsv} {ncsv} {pdump}
# として実行すると、学習後のパラメータを入力して検証する

import sys

import train_cls
import percep_cls
import ab_percep_cls
import svm_cls

mode = 1
if sys.argv[0] == 'chk_training.py':
    mode = 2

method = sys.argv[1]
posi_file = sys.argv[2]
nega_file = sys.argv[3]
param_file = sys.argv[4]

print("mode=%d method=%s" % (mode, method))

if method == 'p':
    perceptron = percep_cls.Perceptron()
    training = train_cls.Training(perceptron)
    if mode == 1:
        training.training(posi_file, nega_file, param_file)
    else:
        training.check(posi_file, nega_file, param_file)

elif method == 'a':
    ab_perceptron = ab_percep_cls.Perceptron()
    training = train_cls.Training(ab_perceptron)
    if mode == 1:
        training.training(posi_file, nega_file, param_file)
    else:
        training.check(posi_file, nega_file, param_file)

elif method == 's':
    svm = svm_cls.SVM()
    training = train_cls.Training(svm)
    if mode == 1:
        training.training(posi_file, nega_file, param_file)
    else:
        training.check(posi_file, nega_file, param_file)
