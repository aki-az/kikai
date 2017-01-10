# -*- coding: utf-8 -*-
# for python3
#
# 特徴抽出した正例・負例の位置関係をグラフ化する
# python chk_feature.py {pcsv} {ncsv} 
# pcsv : 正例のCSVファイル名
# ncsv : 負例のCSVファイル名

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

# Main
if __name__ == '__main__':

    # 特徴量データを入力する
    # 1行が、特徴ベクトルで、行数=件数となる
    # ヘッダはなし
    posiX = pd.read_csv(sys.argv[1], header=None, skiprows=1)
    negaX = pd.read_csv(sys.argv[2], header=None, skiprows=1)

    # 各次元の平均値を計算する
    # 新たにDataFrameを用意して、各次元の平均値を格納する
    # 正例 - 負例の差も計算する
    v = DataFrame()
    v['posi'] = posiX.apply(np.mean, axis=0)
    v['nega'] = negaX.apply(np.mean, axis=0)
    v['diff'] = v['posi'] - v['nega']

    # グラフ描画する
    plt.style.use('ggplot')  # スタイルの設定。なくてもOK
    v.plot()    # 描画する
    plt.show()  # 描画したグラフを表示する

