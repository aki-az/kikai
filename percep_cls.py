# -*- coding: utf-8 -*-
# for python3
# パーセプトロンのクラス

import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

class Perceptron:

#    THRESHOLD = 30000
    THRESHOLD = 50000
    NUM_ITERATION = 100
    
#    def __init__(self, params):

    def set_params(self, params):
        (self.W0, self.W) = params

    # パーセプトロンで予測(分類)する
    # X : 計算対象(numpyの配列)
    def predict(self, X):
    
        # w0, w : バイアス、重み
        t = self.W0 + np.sum(self.W * X)
        if t > self.THRESHOLD:
            return [t]
        else:
            return [-1]

    # パーセプトロンで学習する
    # posiX, negaX : 学習データ(numpy)
    def training(self, posiX, negaX):
    
        # 正例と負例の件数を取得する
        nPosi = posiX.shape[0]
        nNega = negaX.shape[0]
    
        # 正例と負例の教師データを準備する
        yP = np.tile([1], nPosi)
        yN = np.tile([-1], nNega)
        y = np.append(yP, yN)
    
        # 正例と負例をマージしてnumpy配列を作成する
        X = np.array( posiX.append(negaX) )
    
        # 件数と次元数
        n, d = X.shape
    
        # 重みベクトルを初期化する
        W = np.zeros(d, dtype=np.float)
        w0 = 0.0
    
        # w0用のバイアス値を設定する(学習データの平均値)
        bias = X.mean()
        print("bias = %.2f" % bias)
    
        # 重みの変遷の保存場所を確保する
        whist = DataFrame([[w0, 0.0, 0.0]], columns=['w0','W', 'err_rate'])
    
        # 繰り返し学習する
        for i in range(self.NUM_ITERATION):
    
            # 学習データの順をランダムにする
            seq = np.random.permutation(np.array(range(n)))
    
            err, fn, fp = 0, 0, 0
            tw = 0.0
            for j in seq:
    
                # 推定する
                t = w0 + np.sum(W * X[j])
    
                # 推定が誤りの場合、パラメータを修正する
                if y[j] * t <= 0:
                    w0 += y[j] * bias
                    W += y[j] * X[j]
                    tw += np.sum(abs(y[j] * X[j]))
    
                    # 誤り率の推移をみる(学習とは関係はない)
                    err += 1
                    if y[j] == 1: fn += 1
                    else: fp += 1
         
            err_rate = err * 100.0 / n
            fn_rate = fn * 100.0 / nPosi
            fp_rate = fp * 100.0 / nNega
            whist = whist.append( Series([w0, tw, err_rate], ['w0', 'W', 'err_rate']), ignore_index=True)
            print("i=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f w0=%.2f" % (i+1, err_rate, fn_rate, fp_rate, w0))
            if err_rate < 0.5:
                break
    
        print("--------------------")
        return [(w0, W), err_rate, whist]

    # 学習結果を検証する
    # params : (w0, W)
    #   w0 : 重みベクトルのバイアス
    #   W : 重みベクトル
    # posiX, negaX : 検証データ(numpy)
    # spec_str : 検証結果グラフのタイトル文字列
    def chk_training(self, params, posiX, negaX, spec_str):

        # 正例と負例の件数を取得する
        nPosi = posiX.shape[0]
        nNega = negaX.shape[0]
    
        # 正例と負例の教師データを準備する
        yP = np.tile([1], nPosi)
        yN = np.tile([-1], nNega)
        y = np.append(yP, yN)
    
        # 正例と負例をマージしてnumpy配列を作成する
        X = np.array( posiX.append(negaX) )

        (w0, W) = params
        
        n, d = X.shape  # 件数と次元数
        err, fn, fp = 0, 0, 0
    
        p_scores = []
        n_scores = []
    
        for j in range(n):
            t = w0 + np.sum(W * X[j])
            tt = y[j] * t
            if tt <= 0:
                err += 1
                if y[j] == 1: fn += 1
                else: fp += 1
            if y[j] == 1:
                p_scores.append(t)
            else:
                n_scores.append(t)
                
        err_rate = err * 100.0 / n
        fn_rate = fn * 100.0 / nPosi
        fp_rate = fp * 100.0 / nNega
        print("Perceptron check: err=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f" % (err, err_rate, fn_rate, fp_rate))
    
        # グラフ描画の準備をする
        fig = plt.figure()
        subplots1 = fig.add_subplot(1,1,1)
        subplots1.plot(p_scores)
        subplots1.plot(n_scores)
        plt.grid()
        plt.suptitle(spec_str)
        fig.show()
        plt.show()