# -*- coding: utf-8 -*-
#
import sys
import pickle

import sys
import pickle
import numpy as np
import sklearn.svm
import pandas as pd
from pandas import Series, DataFrame

#X,y = pickle.load(open(sys.argv[1], 'rb'))
#print(X.shape[0])
#print(X.shape[1])
#print(y.shape[0])
#classifier = sklearn.svm.LinearSVC(C = 0.0001)
#classifier.fit(X,y)
#pickle.dump(classifier, open(sys.argv[2], 'wb'))

def training_svm(posiX, negaX):

    # 正例と負例の件数を取得する
    nPosi = posiX.shape[0]
    nNega = negaX.shape[0]

    # 正例と負例の教師データを準備する
    yP = np.tile([1], nPosi)
    yN = np.tile([0], nNega)
    y = np.append(yP, yN)

    # 正例と負例をマージしてnumpy配列を作成する
    # (pandas -> numpy)
    X = np.array( posiX.append(negaX) )

    # SVMで学習する
    classifier = sklearn.svm.LinearSVC(C = 0.0001)
    classifier.fit(X,y)

    # 学習結果のパラメータを出力する
    pickle.dump(classifier, open(sys.argv[3], 'wb'))

# Main
if __name__ == '__main__':

    # 特徴量データを入力する
    # 1行が、特徴ベクトルで、行数=件数となる
    # ヘッダはなし
    posiX = pd.read_csv(sys.argv[1], header=None)
    negaX = pd.read_csv(sys.argv[2], header=None)

    # SVMの学習をする
    training_svm(posiX, negaX)


