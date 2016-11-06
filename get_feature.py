# -*- coding: utf-8 -*-
# 画像から特徴抽出する
#
# python get_feature.py {dir} {csv} 
# dir : 画像ディレクトリ
# csv : 出力CSVファイル名

import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
#import pickle
import csv

WIDTH, HEIGHT = (64, 64)
cell_size = 4
LBP_POINTS = 24
LBP_RADIUS = 3
#cell_size = 8
#LBP_POINTS = 8
#LBP_RADIUS = 2

bins = LBP_POINTS + 2

# LBPからヒストグラムを作成する
# 入力データをセル単位に分割し、LBPの値ごとに出現数を数える
# 入力: 2次元のLBPデータ
# 出力: 1次元のリスト
def get_histogram_feature(lbp):
    histograms = []
    for y in range(0, HEIGHT, cell_size):
        for x in range(0, WIDTH, cell_size):

            # LBPの値の種類分の格納場所を作成する
            h = np.zeros(shape = (bins,))

            # セル内のヒストグラムを作成する
            for dy in range(cell_size):
                for dx in range(cell_size):
                    h[ int(lbp[y + dy, x + dx]) ] += 1

            # histogramsにリスト(h)を追加する
            # (histogramsはリストのリストになる)
            histograms.append(h)
    return np.concatenate(histograms)

def get_features(directory, writer):
    n = 0
    for fn in iglob('%s/*.png' % directory):

        image = color.rgb2gray(io.imread(fn))

        # 画像からLBPを取得する
        lbp_image = feature.local_binary_pattern(
            image, LBP_POINTS, LBP_RADIUS, 'uniform')

        # LBPのヒストグラムを作成する
        h = get_histogram_feature(lbp_image)

        # CSV出力する
        writer.writerow(h)
        n += 1
    return n

def main():
    img_dir = sys.argv[1]

    fpo = open(sys.argv[2], 'w')
    writer = csv.writer(fpo)

    n = get_features(img_dir, writer)

    fpo.close()

    print("num=%d" % n)

if __name__ == "__main__":
    main()

