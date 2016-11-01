# -*- coding: utf-8 -*-

import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
import pickle

WIDTH, HEIGHT = (64, 64)
LBP_POINTS = 24
LBP_RADIUS = 3
#LBP_POINTS = 8
#LBP_RADIUS = 2
cell_size = 4

def get_histogram_feature(lbp):
    bins = LBP_POINTS + 2
    histograms = []
    for y in range(0, HEIGHT, cell_size):
        for x in range(0, WIDTH, cell_size):
            histogram = np.zeros(shape = (bins,))
            for dy in range(cell_size):
                for dx in range(cell_size):
                    histogram[lbp[y + dy, x + dx]] += 1
            histograms.append(histogram)
    #print(histograms)
    # histogramsは2次元配列
    # cell_size=4,LBP=24なら、(24+2)x(16x16)の配列になる
    return np.concatenate(histograms)

def get_features(directory):
    features = []
    for fn in iglob('%s/*.png' % directory):
        #print fn
        image = color.rgb2gray(io.imread(fn))
        lbp_image = feature.local_binary_pattern(
            image, LBP_POINTS, LBP_RADIUS, 'uniform')
        features.append(get_histogram_feature(lbp_image))
        #print("###", len(features), len(features[0]), "###")
    return features

def main():
    positive_dir = sys.argv[1]
    negative_dir = sys.argv[2]
    positive_samples = get_features(positive_dir)
    negative_samples = get_features(negative_dir)
    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)
    X = np.array(positive_samples + negative_samples)
    y = np.array([1 for i in range(n_positives)] +
                 [0 for i in range(n_negatives)])
    pickle.dump((X, y), open(sys.argv[3], 'wb'))
    print("posi=%d nega=%d" % (n_positives, n_negatives))

if __name__ == "__main__":
    main()

