# -*- coding: utf-8 -*-
# for python3
# AdaBoost+パーセプトロンのクラス

import math
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

class AB_Perceptron:

    #THRESHOLD = 20000
    THRESHOLD = 250000
    NUM_ITERATION = 30
    NUM_DETECTORS = 3
    
#    def __init__(self, params):

    def set_params(self, params):
        self.params = params

    # AdaBoost + パーセプトロンでデータを分類する
    # X : 計算対象(numpyの配列)
    def predict_old(self, X):
    
        ts = []
    
        tt = 0
        for i in range(len(self.params)):
            (alpha, w0, W) = self.params[i]
     
            # w0, w : バイアス、重み
            t = w0 + np.sum(W * X)
    
    #        if t > 0:
            if t > self.THRESHOLD: tt = 1.0
            else: tt -1.0
            ts.append(alpha * tt)
    
        # 各検出器の結果を総括して判定する
        # とりあえず多数決を取る
    #    print("adaboost detectors:"); print(ts)
        pn = sum(ts)
        if pn > 0: return [1]
        else: return [-1]
    
    def predict(self, X):
    
        tt = 0
        for i in range(len(self.params)):
            (alpha, w0, W) = self.params[i]
     
            # w0, w : バイアス、重み
            t = w0 + W.dot( X )
            tt += alpha * t
    
        if tt > self.THRESHOLD: return [tt]
        else: return [-1]


    # Perceptronのアルゴリズム（確率的勾配降下法）を実行
    # X : 学習データ(numpy)
    # y : 教師データ(numpy)
    # wd : 学習データの重み(numpy)
    # nPosi, nNega : 正例、負例の件数
    def training_perceptron(self, X, y, wd, nPosi, nNega):
    
        # 学習データの件数と次元数
        n, d = X.shape
    
        # 学習データの重みを正規化する
        # 最大値を1.0にする
        # 弱識別器の重みベクトル修正量を学習データの重みに合わせて
        # 調節するが、修正量を最大100%にするための処置
        wdp = wd / wd.max()
        print("wd.max() = %f" % wd.max())
    
        # 弱識別器の重みベクトルを初期化する
        W = np.zeros(d, dtype=np.float)
        w0 = 0.0
    
        # w0用のバイアス値を設定する(学習データの平均値)
        bias = X.mean()
        print("bias = %.2f" % bias)
    
        # 弱識別器の重みの変遷の保存場所を確保する
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
    
                # 推定が誤りの場合、重みベクトルを修正する
                # 学習データの重み(wdp)に合わせて、修正量を調節する
                if y[j] * t <= 0:
                    w0 += y[j] * bias * wdp[j]
                    W += y[j] * X[j] * wdp[j]
                    tw += np.sum(abs(y[j] * X[j] * wdp[j]))
    
                    # 誤り率の推移をみる(学習とは関係はない)
                    err += 1
                    if y[j] == 1: fn += 1
                    else: fp += 1
    
            err_rate = err * 100.0 / n
            fn_rate = fn * 100.0 / nPosi
            fp_rate = fp * 100.0 / nNega
            whist = whist.append( Series([w0, tw, err_rate], ['w0', 'W', 'err_rate']), ignore_index=True)
            print("i=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f w0=%.2f W(mean)=%.2f" % (i+1, err_rate, fn_rate, fp_rate, w0, W.mean()))
            
            if err_rate < 1:
                break
    
        return [w0, W, err_rate, whist]
    
    
    # 弱識別器の重みを計算する
    # ついでに、計算した重みで学習データの重みを更新する
    # w0 : 識別器の重みベクトルのバイアス
    # W : 識別器の重みベクトル
    # X : 学習データの配列(np.array)
    # y : 教師データ(np.array)
    # wd : 学習データの重み(np.array)
    def get_alpha(self, w0, W, X, y, wd):
    
        n, d = X.shape
        wd_errsum = 0.0    # 推定を誤ったデータの重み合計
        err_X = []    # 推定を誤ったデータの番号
    
        # 推定を誤った学習データの重みを集計して、学習データの番号を記録する
        for j in range(n):
    
            # 推定する
            t = w0 + np.sum(W * X[j])
    
            # 推定を誤った学習データの重みを足し合わせる
            if y[j] * t <= 0:
                wd_errsum += wd[j]
                err_X.append(j)
    
        # 誤り率を計算する
        err_rate = wd_errsum / wd.sum()
    
        # 識別器の重要度を計算する
        alpha = 1.0
        if err_rate > 0.0:
            alpha = math.log((1.0 - err_rate) / err_rate)
        print("get_alpha() alpha=%f num_err=%d" % (alpha, len(err_X)))
    
        # 学習データの重みを更新する
        # 推定を誤ったデータが更新対象となる
        for idx in err_X:
            wd[idx] = wd[idx] * math.exp(alpha)
    
        return [alpha, wd]

    # AdaBoost+パーセプトロンで学習する
    # 弱識別器としてパーセプトロンを使用する
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
    
        # 弱識別器の保存場所
        params = []
    
        # 学習データの重みを準備する(一律の値)
        wd = np.tile([1.0/n], n)  # 総和が1.0になる
    
        # 弱識別器を何個か作成する
        for i in range(self.NUM_DETECTORS):
            print("----\nAdaBoost m=%d" % (i+1))
    
            # パーセプトロンで学習する
            # 重み付きの教師データで学習し、弱識別器の重みベクトルを得る
            w0, W, err_rate, whist = self.training_perceptron(X, y, wd, nPosi, nNega)
    
            # 弱識別器の重みを計算する
            # ついでに、計算した重みで学習データの重みを更新する
            alpha, wd = self.get_alpha(w0, W, X, y, wd)
    
            # 弱識別器の重要度と重みベクトルを保存する
            params.append( [alpha, w0, W] )
    
        return [params, err_rate, whist]

    # 学習結果を検証する
    # params : (w0, W)
    #   w0 : 重みベクトルのバイアス
    #   W : 重みベクトル
    # posiX, negaX : 検証データ(numpy)
    # spec_str : 検証結果グラフのタイトル文字列
    def chk_training(self, params, posiX, negaX, spec_str):
        
        nd = len(params)  # 弱識別器の数
        for i in range(nd):
            print("alpha=%f" % params[i][0])

        # 正例と負例の件数を取得する
        nPosi = posiX.shape[0]
        nNega = negaX.shape[0]
    
        # 正例と負例の教師データを準備する
        yP = np.tile([1], nPosi)
        yN = np.tile([-1], nNega)
        y = np.append(yP, yN)
    
        # 正例と負例をマージしてnumpy配列を作成する
        X = np.array( posiX.append(negaX) )
    
        n, d = X.shape  # 件数と次元数
        err, fn, fp = 0, 0, 0
    
        p_scores = []
        n_scores = []
        for j in range( nd+1 ):
            p_scores.append([])
            n_scores.append([])
    
        # 学習データを順に推定する
        for i in range(n):
    
            # 弱識別器に順番に推定させる
            tt = 0.0
            for j in range( nd ):
    
                # 弱識別器のパラメータを設定する
                alpha = params[j][0]
                w0 = params[j][1]
                W = params[j][2]
    
                # 推定する
                t = w0 + np.sum(W * X[i])
    
                # 推定結果(1 or -1)に識別器の重みをかけたものを足し合わせる
#                if t > 0: tt += alpha
#                else: tt -= alpha
                tt += alpha * t

                if y[i] == 1:
                    p_scores[j].append(t)
                else:
                    n_scores[j].append(t)

            if y[i] * tt <= 0:
                err += 1
                if y[i] == 1: fn += 1
                else: fp += 1
    
            if y[i] == 1:
                p_scores[nd].append(tt)
            else:
                n_scores[nd].append(tt)
    
    
        err_rate = err * 100.0 / n
        fn_rate = fn * 100.0 / nPosi
        fp_rate = fp * 100.0 / nNega
        print("AdaBoost check: err=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f" % (err, err_rate, fn_rate, fp_rate))

        # グラフ描画の準備をする
        fig = plt.figure()
        
        for i in range(nd+1):
            t_sobplot = fig.add_subplot(nd+1, 1, i+1)
            if i < nd:
                t_sobplot.set_title("detector %d" % (i+1))
            else:
                t_sobplot.set_title("integrated detector")
            t_sobplot.plot(p_scores[i])
            t_sobplot.plot(n_scores[i])
            t_sobplot.grid()
        
        plt.suptitle(spec_str)
        fig.show()
        plt.show()        
