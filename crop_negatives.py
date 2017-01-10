# -*- coding: utf-8 -*-
# for python3
# Original https://github.com/t-abe/cat-face-detection
# 負例画像を作成する
# 入力画像からランダムに画像の一部を切り出して、ファイル出力する
#
import sys
import numpy as np
from skimage import io, transform, img_as_ubyte
import warnings
from glob import glob
import random

def main():
    if len(sys.argv) < 4:
        print("./crop_negatives.py INPUT_DIR OUTPUT_DIR N [prefix]")
        return
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    n_negatives = int(sys.argv[3])
    prefix = ""
    if len(sys.argv) >= 5:
        prefix = sys.argv[4]

    # 入力画像のパスを準備する
    image_list = glob('%s/*.jpg' % input_dir) \
        + glob('%s/*.JPG' % input_dir) \
        + glob('%s/*.png' % input_dir)

    count = 0
    for i, image_path in enumerate(image_list):
        print(image_path)
        image = io.imread(image_path)
        assert image.shape[0] >= 64 or image.shape[1] >= 64
        
        # 分かりにくい。
        # n_negatives枚の画像を得るために、1つの入力画像から
        # 必要な数だけ画像を切り出している。
        j = 0
        while n_negatives * (i + 1) / len(image_list) > count:
            cropped = crop_randomly(image)
#            io.imsave('%s/%d.png' % (output_dir, count), cropped)
            if save_img('%s/%s%d.png' % (output_dir, prefix, count), cropped) == 0:
                count += 1
                j = 0
            else:
                j += 1
                if j == 3:
                    break

def save_img(path, image):
    ret = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            io.imsave(path, image)
    except:
        ret = -1
        print("low contrast image: ",path)

    return ret

# 入力画像からランダムに部分画像を切り出す
# 64x64のサイズを切り出すが、入力画像が大きい場合は、
# 小さい範囲を切り出すと、コントラストの低い画像になるので、
# 入力画像を縮小してから切りだす。
def crop_randomly(image):
    h, w, _ = image.shape

    # 入力画像が大きい場合は縮小する
    size = max(64, w / 16, h / 16)
    if size > 64:
        image = transform.rescale(image, 64/size)

        # rescale()を使うと、値のスケール(型)が
        # 0-255(uint)から0.0-1.0(float)に変換されるので元に戻す。
        # その際、情報落ちのwarnが出るので抑止する。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = img_as_ubyte(image)

        w = int(w * 64/size)
        h = int(h * 64/size)

    x = random.randint(0, w - 64)
    y = random.randint(0, h - 64)
    cropped = image[y:y + 64, x:x + 64]
    return cropped


def crop_randomly_old(image):
    h, w, _ = image.shape
    x = random.randint(0, w - 64)
    y = random.randint(0, h - 64)
    cropped = image[y:y + 64, x:x + 64]
    return cropped

if __name__ == "__main__":
    main()
