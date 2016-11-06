# -*- coding: utf-8 -*-
#
# パーセプトロンで学習する
# python train_percep.py {pcsv} {ncsv} {pdump}
# pcsv : 正例のCSVファイル名
# ncsv : 負例のCSVファイル名
# pdump : 学習後のパラメータ

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import multivariate_normal

# Perceptronのアルゴリズム（確率的勾配降下法）を実行
# posiX,negaX : pandas DataFrame
def training_perceptron(posiX, negaX):

    # 正例と負例の件数を取得する
    nPosi = posiX.shape[0]
    nNega = negaX.shape[0]

    # 正例と負例の教師データを準備する
    yP = np.tile([1], nPosi)
    yN = np.tile([-1], nNega)
    y = np.append(yP, yN)

    # 正例と負例をマージしてnumpy配列を作成する
    # (pandas -> numpy)
    X = np.array( posiX.append(negaX) )
    n, d = X.shape  # 件数と次元数

    # 重みベクトルを初期化する
    W = np.zeros(d, dtype=np.float)
    w0 = 0.0

    # w0用のバイアス値を設定する(学習データの平均値)
    bias = X.mean()
    print("bias = %.2f" % bias)

    # 重みの変遷の保存場所を確保する
    whist = DataFrame([[w0, 0.0, 0.0]], columns=['w0','w', 'err_rate'])

    # Iterationを何回か実施
    for i in range(10):

        # 学習データの順をランダムにする
        seq = np.random.permutation(np.array(range(n)))

        err, fn, fp = 0, 0, 0
        tw = 0.0
        for j in seq:

            # 推定する
            t = w0
            for k in range(d):
                t += W[k] * X[j,k]

            # 推定が誤りの場合、パラメータを修正する
            if y[j] * t <= 0:
                w0 += y[j] * bias
                for k in range(d):
                    W[k] += y[j] * X[j,k]
                    tw += abs(y[j] * X[j,k])

                # 誤り率の推移をみる(学習とは関係はない)
                err += 1
                if y[j] == 1: fn += 1
                else: fp += 1
     
        err_rate = err * 100.0 / n
        fn_rate = fn * 100.0 / nPosi
        fp_rate = fp * 100.0 / nNega
        whist = whist.append( Series([w0, tw, err_rate], ['w0', 'w', 'err_rate']), ignore_index=True)
        print("i=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f w0=%.2f" % (i+1, err_rate, fn_rate, fp_rate, w0))

    print("--------------------")

    # 判定誤差を計算
    err, fn, fp = 0, 0, 0
    for j in range(n):
        t = w0
        for k in range(d):
            t += W[k] * X[j,k]
        if y[j] * t <= 0:
            err += 1
            if y[j] == 1: fn += 1
            else: fp += 1
    err_rate = err * 100.0 / n
    fn_rate = fn * 100.0 / nPosi
    fp_rate = fp * 100.0 / nNega
    print("check: err=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f" % (err, err_rate, fn_rate, fp_rate))

    return [w0, W, err_rate, whist]

# Main
if __name__ == '__main__':

    # 特徴量データを入力する
    # 1行が、特徴ベクトルで、行数=件数となる
    # ヘッダはなし
    posiX = pd.read_csv(sys.argv[1], header=None)
    negaX = pd.read_csv(sys.argv[2], header=None)

    # パーセプトロンの学習をする
    ret = training_perceptron(posiX, negaX)
    w0 = ret[0]
    w = ret[1]
    err_rate = ret[2]
    paramhist = ret[3]

    # 学習結果のパラメータを出力する
    pickle.dump((w0, w), open(sys.argv[3], 'wb'))

    # グラフ描画の準備をする
    fig = plt.figure()
    subplots1 = fig.add_subplot(2,2,1)
    subplots2 = fig.add_subplot(2,2,2+1)

    # 結果の表示
    print("ERR %.2f%%" % err_rate)
#    print("W ", w)
#    paramhist.plot(ax=subplots2)
    paramhist['w'].plot(ax=subplots1)
    paramhist['w0'].plot(ax=subplots2)
    paramhist['err_rate'].plot(ax=subplots2)
    subplots2.legend(loc=1)

    fig.show()
    plt.show()
