# -*- coding: utf-8 -*-
# for python3
#
# いろいろな手法で画像検出する
# python detect_img.py {m} {param} {img} 
# m    : 'p'=パーセプトロン, 'a'=AdaBoost+パーセプトロン, 's'=SVM
# param : 学習後のパラメータファイル
# img : 画像ファイル

import sys

import detect_img_cls
import percep_cls
import ab_percep_cls
import svm_cls


method = sys.argv[1]
param_file = sys.argv[2]
img_file = sys.argv[3]

if method == 'p':
    c = percep_cls.Perceptron()
    detect = detect_img_cls.DetectImg(c.predict, param_file)

    params = detect.get_params()
    c.set_params(params)

    detect.detect(img_file, "Perceptron")

elif method == 'a':
    c = ab_percep_cls.AB_Perceptron()
    detect = detect_img_cls.DetectImg(c.predict, param_file)

    params = detect.get_params()
    c.set_params(params)

    detect.detect(img_file, "AdaBoost + Perceptron")

elif method == 's':
    c = svm_cls.SVM()
    detect = detect_img_cls.DetectImg(c.predict, param_file)

    params = detect.get_params()
    c.set_params(params)

    detect.detect(img_file, "SVM")
