# -*- coding: utf-8 -*-
# for python3
# 画像から特徴抽出する
# Original https://github.com/t-abe/cat-face-detection
#
# python get_feature.py {spec} {dir} {csv} [P]
# spec: n1:n2:n3 = cell_size,LBP_RADIUS,LBP_POINTS
#   4:3:24, 4:2:8, 8:3:24, 8:2:8
# dir : 画像ディレクトリ
# csv : 出力CSVファイル名
# P : 画像を水増しする(画像を反転させた特徴量を追加する)

import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
#import pickle
import csv

WIDTH, HEIGHT = (64, 64)

# コマンド引数からパラメータを取得する
n1,n2,n3 = sys.argv[1].split(":")
cell_size,LBP_RADIUS,LBP_POINTS = int(n1), int(n2), int(n3)
bins = LBP_POINTS + 2
img_dir = sys.argv[2]
csv_file = sys.argv[3]

Pad_mode = 0    # 水増しモード
if len(sys.argv) >= 5 and sys.argv[4] == "P":
    Pad_mode = 1

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

        # 水増しモードがオンの場合、左右反転画像の特徴を出力する
        if Pad_mode == 1:
            lbp_image = feature.local_binary_pattern(
                np.fliplr(image), LBP_POINTS, LBP_RADIUS, 'uniform')
            h = get_histogram_feature(lbp_image)
            writer.writerow(h)
            n += 1

    return n

def main():

    # 特徴量のcsv出力準備をする
    fpo = open(csv_file, 'w')
    writer = csv.writer(fpo)

    # セルサイズなどを先頭に出力する
    writer.writerow(['LBP', 1.0, cell_size, LBP_RADIUS, LBP_POINTS])

    # 特徴抽出する
    n = get_features(img_dir, writer)

    fpo.close()

    print("num=%d" % n)

if __name__ == "__main__":
    main()

