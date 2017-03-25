# -*- coding: utf-8 -*-
# for python3
# 学習クラス

import pickle
import matplotlib.pyplot as plt
import pandas as pd

class Training:

    NUM_ITERATION = 100

    def __init__(self, train_cls):
        self.train_cls = train_cls
#        self.training = train_cls.training
#        self.chk_training = train_cls.chk_training

    def train(self, mode, posi_file, nega_file, param_file):

        # 検証モードの場合は、学習結果のパラメータを入力する
        if mode == 2:
            d1,d2,d3,params = pickle.load(open(param_file, "rb"))    # d1,d2,d3 are dummy
            print("parameter spec: cell_size=%d LBP_radius=%d" % (d1, d2))
    
        # 特徴量データを入力する
        # 1行が、特徴ベクトルで、行数=件数となる
        # ヘッダはなし
        # 先頭行は特徴抽出のパラメータで、2行目以降と内容が異なるので個別に入力する
        fparam = pd.read_csv(posi_file, header=None, nrows=1)
        cell_size = fparam[2].tolist()[0]
        LBP_RADIUS = fparam[3].tolist()[0]
        LBP_POINTS = fparam[4].tolist()[0]
        spec_str = "input features spec: cell_size=%d LBP_radius=%d" % (cell_size, LBP_RADIUS)
        print(spec_str)

        posiX = pd.read_csv(posi_file, header=None, skiprows=1)
        negaX = pd.read_csv(nega_file, header=None, skiprows=1)
    
        # 正例と負例の件数を取得する
        nPosi = posiX.shape[0]
        nNega = negaX.shape[0]
        print("number of data: posi=%d nega=%d" % (nPosi, nNega))
    
        # 検証モード
        if mode == 2:
            self.train_cls.chk_training(params, posiX, negaX, spec_str)
            return
    
        # 学習をする
        ret = self.train_cls.training(posiX, negaX)
        params = ret[0]
        err_rate = ret[1]
        paramhist = ret[2]
    
        # 学習結果を検証する
        self.train_cls.chk_training(params, posiX, negaX, spec_str)
    
        # 特徴抽出のパラメータと、学習結果のパラメータを出力する
        pickle.dump([cell_size,LBP_RADIUS,LBP_POINTS,params], open(param_file, 'wb'))
    
        # グラフ描画の準備をする
        fig = plt.figure()
        subplots1 = fig.add_subplot(2,1,1)
        subplots2 = fig.add_subplot(2,1,2)
    
        # 結果の表示
        if err_rate is not None:
            print("ERR %.2f%%" % err_rate)
        #    print("W ", W)
        #    paramhist.plot(ax=subplots2)
            paramhist['W'].plot(ax=subplots1)
            paramhist['w0'].plot(ax=subplots2)
            paramhist['err_rate'].plot(ax=subplots2)
            subplots1.legend(loc=1)
            subplots2.legend(loc=1)
        
            fig.show()
            plt.show()


    def training(self, posi_file, nega_file, param_file):
        self.train(1, posi_file, nega_file, param_file)

    def check(self, posi_file, nega_file, param_file):
        self.train(2, posi_file, nega_file, param_file)
