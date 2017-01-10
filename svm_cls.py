# -*- coding: utf-8 -*-
# for python3
# SVMのクラス

import numpy as np
import sklearn.svm
import matplotlib.pyplot as plt

class SVM:

    THRESHOLD = 1.8
    
#    def __init__(self, params):

    def set_params(self, params):
        self.params = params

    # SVMで予測(分類)する
    # X : 計算対象(numpyの配列)
    def predict(self, X):

        t = self.params.decision_function(X)
    
        if t > self.THRESHOLD:
            return [t]
        else:
            return [-1]

    # SVMで学習する
    # posiX, negaX : 学習データ(numpy)
    def training(self, posiX, negaX):
    
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
    
        return [classifier, None, None]

    # 学習結果を検証する
    # params : (w0, W)
    #   w0 : 重みベクトルのバイアス
    #   W : 重みベクトル
    # posiX, negaX : 検証データ(配列)
    # spec_str : 検証結果グラフのタイトル文字列
    def chk_training(self, classifier, posiX, negaX, spec_str):

        # 正例と負例の件数を取得する
        nPosi = posiX.shape[0]
        nNega = negaX.shape[0]

        # numpyに変換する
        XP = np.array( posiX )
        XN = np.array( negaX )

        p_scores = []
        n_scores = []
        err, fn, fp = 0, 0, 0
        
        for i in range(nPosi):
            t = classifier.decision_function(XP[i].reshape(1, -1))
            p_scores.append(t)
            if t < 0:
                err += 1
                fn += 1

        for i in range(nNega):
            t = classifier.decision_function(XN[i].reshape(1, -1))
            n_scores.append(t)
            if t < 0:
                err += 1
                fp += 1
                
        err_rate = err * 100.0 / (nPosi + nNega)
        fn_rate = fn * 100.0 / nPosi
        fp_rate = fp * 100.0 / nNega
        print("SVM check: err=%d err_rate=%.1f fn_rate=%.1f fp_rate=%.1f" % (err, err_rate, fn_rate, fp_rate))
    
        # グラフ描画の準備をする
        fig = plt.figure()
        subplots1 = fig.add_subplot(1,1,1)
        subplots1.plot(p_scores)
        subplots1.plot(n_scores)
        plt.grid()
        plt.suptitle(spec_str)
        fig.show()
        plt.show()
