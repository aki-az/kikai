# for python3
# Original https://github.com/t-abe/cat-face-detection
# 猫画像から顔部分を切り出した画像を作成する
# 猫画像とセットになっているアノテーションを入力し、それをもとに顔部分を切り出す。
# オプションで、負例画像の出力もできる。
# 顔以外の部分をランダムに切り出す。

import sys
import numpy as np
from skimage import io, transform, img_as_ubyte
import warnings
from glob import iglob
import random

Max_sizex = 0
Max_sizey = 0

def main():
    if len(sys.argv) < 3:
        print("crop_faces.py INPUT_DIR OUTPUT_DIR [OUTPUT2_DIR]")
        return
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_dir_n = ""
    if len(sys.argv) >= 4:
        output_dir_n = sys.argv[3]

    for i, image_path in enumerate(iglob('%s/*/*.jpg' % input_dir)):
        annotation_path = '%s.cat' % image_path
        try:
            # アノテーションファイルを入力して、分解する
            annotation = parse_annotation(open(annotation_path).read())
        except:
            continue
        
        # 画像ファイルを入力して、顔部分を切り出す
        face = crop_face(io.imread(image_path), annotation)
        if face is None: continue

        # 切り出した画像を保存する前に型変換する
        # そのままio.imsave()すると、警告がうるさいので。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = img_as_ubyte(face)
            
        # 切り出した画像を保存する
        io.imsave('%s/%d.png' % (output_dir, i), img)
        print("img[%s] posi[%d]" % (image_path, i))

        # 負例画像も切り出す
        if output_dir_n == "":
            continue
            
        face = crop_face_n(io.imread(image_path), annotation)
        if face is None:
            continue
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = img_as_ubyte(face)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                io.imsave('%s/%d.png' % (output_dir_n, i), img)
            #print("img[%s] nega[%d]" % (image_path, i))
        except:
            print("low contrast image: ",image_path)


# アノテーション情報を分解する
def parse_annotation(line):

    # アノテーション情報を行単位に区切る
#    v = map(int, line.split())      # python 2
    v = list(map(int, line.split())) # python 3

    # アノテーション格納場所を初期化
    ret = {}
    parts = ["left_eye", "right_eye", "mouth",
             "left_ear1", "left_ear2", "left_ear3",
             "right_ear1", "right_ear2", "right_ear3"]

    # アノテーション情報をx,yごとに取り出して、ret(辞書型)に格納する
    # アノテーションは19個の数字列で、先頭は情報の個数(=要素数 / 2)
    # 2個づつ取り出せば良いよう。
    for i, part in enumerate(parts):
        if i >= v[0]: break
        ret[part] = np.array([v[1 + 2 * i], v[1 + 2 * i + 1]])

    return ret

# 画像データから顔部分を取り出す
# image : 画像データ
# an : アノテーション(辞書型)
def crop_face(image, an):
    
    # 顔の中心部分(両目と口の間)を計算する
    diff_eyes = an["left_eye"] - an["right_eye"]
    if diff_eyes[0] == 0 or abs(float(diff_eyes[1]) / diff_eyes[0]) > 0.5:
        return None
    center = (an["left_eye"] + an["right_eye"] + an["mouth"]) / 3
    
    # 中心が口より下にある(つまり逆さま)のは除外
    if center[1] > an["mouth"][1]: return None
        
    # 顔を切り出す枠の位置を計算する
    radius = np.linalg.norm(diff_eyes) * 1.1
    xu = int(center[0] - radius)    # 左端
    xl = int(center[0] + radius)    # 右端
    yu = int(center[1] - radius)    # 上端
    yl = int(center[1] + radius)    # 下端
    if xu < 0 or yu < 0: return None
    if xl > image.shape[1] or yl > image.shape[0]:
        return None
        
    # 顔部分を切り出す
    cropped = image[yu:yl, xu:xl]

    # 切り出した顔の最大サイズを求める
    # (個々の切り出し処理には使用しない)
    global Max_sizex, Max_sizey
    Max_sizex = max(Max_sizex, xl - xu)
    Max_sizey = max(Max_sizey, yl - yu)
    
    return transform.resize(cropped, (64, 64))  # resize()の戻りはfloat型


# 画像データから顔部分以外を取り出す
# 使い方はcrop_face()と同じ
def crop_face_n(image, an):
    
    # 顔の中心部分(両目と口の間)を計算する
    diff_eyes = an["left_eye"] - an["right_eye"]
    if diff_eyes[0] == 0 or abs(float(diff_eyes[1]) / diff_eyes[0]) > 0.5:
        return None
    center = (an["left_eye"] + an["right_eye"] + an["mouth"]) / 3

    # 中心が口より下にある(つまり逆さま)のは除外
    if center[1] > an["mouth"][1]: return None

    # 顔を切り出す枠の位置を計算する
    radius = np.linalg.norm(diff_eyes) * 1.1
    xu = int(center[0] - radius)    # 左端
    xl = int(center[0] + radius)    # 右端
    yu = int(center[1] - radius)    # 上端
    yl = int(center[1] + radius)    # 下端
    if xu < 0 or yu < 0: return None
    if xl > image.shape[1] or yl > image.shape[0]:
        return None

    # 顔部分のサイズ
    size_x = xl -xu
    size_y = yl -yu

    # 以下は負例画像の作成処理
    # 顔と同じサイズで顔以外の場所をランダムに切り出して負例とする

    cropped_n = None
    i = 0
    # ランダムの範囲取得で例外が発生したら、リトライする
    while i < 3:
        m = random.randint(1, 4)  # 切り出す場所を選ぶ
    
        try:
            # 顔部分より左側の領域をランダムに切り出す
            if m == 1:
                x = random.randint(0, xu - size_x)
                y = random.randint(0, image.shape[0] - size_y)
                cropped_n = image[y:y+size_y, x:x+size_x]
        
            # 顔部分より右側の領域をランダムに切り出す
            if m == 2:
                x = random.randint(xl, image.shape[1] -size_x)
                y = random.randint(0, image.shape[0] - size_y)
                cropped_n = image[y:y+size_y, x:x+size_x]
        
            # 顔部分より上側の領域をランダムに切り出す
            if m == 3:
                x = random.randint(0, image.shape[1] - size_x)
                y = random.randint(0, yu -size_y)
                cropped_n = image[y:y+size_y, x:x+size_x]
        
            # 顔部分より下側の領域をランダムに切り出す
            else:
                x = random.randint(0, image.shape[1] - size_x)
                y = random.randint(yl, image.shape[0] -size_y)
                cropped_n = image[y:y+size_y, x:x+size_x]

            return transform.resize(cropped_n, (64, 64))
            
        except:
            i += 1

    return None


if __name__ == "__main__":
    main()
    print("max size x=%d y=%d" % (Max_sizex, Max_sizey))
