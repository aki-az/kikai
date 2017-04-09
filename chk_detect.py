# 学習結果のパラメータを用いて、検出能力を検証する
#
# とりあえずパーセプトロン用に作成した。
# 標準入力から猫画像ファイルリストを読み込み、
# 対になるアノテーションから顔の位置を取得する。
# 次に顔検出を行い、検出結果をチェックする。
#
# 検出結果は"第一候補"、"第二候補以降"、"検出できず"に分類する。
# 検出結果が正しいことの判定は、アノテーションから取得した
# 顔の位置との重なり具合で行う。
#
# python chk_detect.py {param} < {files}
# param : パーセプトロンのパラメータファイル
# files : 画像ファイルのリスト

import sys
import numpy as np
from skimage import io

import detect_img_cls
import percep_cls

Order = 5    # 検出結果の何位までを検出したとみなすか
Overlap_Ratio = 0.6    # 正解の位置との重なり具合(0.0 - 1.0)

# アノテーション情報を分解する
# line : アノテーションデータ
def parse_annotation(line):

    # アノテーション情報を行単位に区切る
    v = list(map(int, line.split())) # python 3

    # アノテーション格納場所を初期化
    ret = {}
    parts = ["left_eye", "right_eye", "mouth",
             "left_ear1", "left_ear2", "left_ear3",
             "right_ear1", "right_ear2", "right_ear3"]

    for i, part in enumerate(parts):
        if i >= v[0]: break
        ret[part] = np.array([v[1 + 2 * i], v[1 + 2 * i + 1]])

    return ret

# アノテーションから顔の位置を求める
# image : 画像データ
# an : アノテーション(辞書型)
def get_facepos(image, an):
    
    # 顔の中心部分(両目と口の間)を計算する
    # 傾きがきつい時はエラーとする
    diff_eyes = an["left_eye"] - an["right_eye"]
    if diff_eyes[0] == 0 or abs(float(diff_eyes[1]) / diff_eyes[0]) > 1.0:
        print("slope error s=%f" % abs(float(diff_eyes[1]) / diff_eyes[0]))
        return None
    center = (an["left_eye"] + an["right_eye"] + an["mouth"]) / 3
    
    # 中心が口より下にある(つまり逆さま)のは除外
    if center[1] > an["mouth"][1]:
        print("upside down error")
        return None
        
    # 顔を切り出す枠の位置を計算する
    radius = np.linalg.norm(diff_eyes) * 1.1
    xu = int(center[0] - radius)    # 左端
    xl = int(center[0] + radius)    # 右端
    yu = int(center[1] - radius)    # 上端
    yl = int(center[1] + radius)    # 下端
    if xu < 0 or yu < 0 or xl > image.shape[1] or yl > image.shape[0]:
        print("range error")
        return None
    return {'x':xu, 'y':yu, 'width':(xl -xu), 'height':(yl -yu)}

# 検出能力を検証する
# param_file : パラメータファイル
# order : 検出結果の何位までを検出したとみなすか
# overlap_ratio : 正解の位置との重なり具合(0.0 - 1.0)
def chk_detect(param_file, order, overlap_ratio):

    # パーセプトロンのインスタンスを準備する
    c = percep_cls.Perceptron()
    detect = detect_img_cls.DetectImg(c.predict, param_file)
    params = detect.get_params()
    c.set_params(params)
    
    num = 0
    hit,hit2,nohit,errcnt = 0,0,0,0
    
    while 1:
      try:
        # 標準入力から画像ファイル名を入力する
        img_file = input()
        print(img_file)
        
        # 画像ファイルと紐づくアノテーションを入力する
        annotation_path = '%s.cat' % img_file
        annotation = parse_annotation(open(annotation_path).read())
        
        # アノテーションから顔の位置を求める
        answer = get_facepos(io.imread(img_file), annotation)
        if answer is None:
            print("skip file ", img_file)
            continue
        num += 1

        # 検出する
        res = 0
        res = detect.detect_noplot(img_file, answer, order, overlap_ratio)
#        detect.detect(img_file, '', answer)

        # 検出結果をカウントする
        print("####### num=%d result=%d" % (num, res))
        if res == 1: hit += 1
        elif res >1:  hit2 += 1
        elif res == 0: nohit += 1
        else: errcnt += 1
        print("num=%d  hit=%d  hit2=%d  nohit=%d  errcnt=%d" % (num,hit,hit2,nohit,errcnt))
        
      except EOFError:
        break
    

# Main
if __name__ == '__main__':
    param_file = sys.argv[1]
    chk_detect(param_file, Order, Overlap_Ratio)    
