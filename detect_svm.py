# -*- coding: utf-8 -*-
#
# SVMによる画像検出
# python detect_svm.py {pdump} {img} 
# pdump : 学習後のパラメータ
# img : 画像ファイル

import sys
import numpy as np
from skimage import io, feature, color, transform
#from glob import iglob
import pickle
import matplotlib.pyplot as plt

WIDTH, HEIGHT = (64, 64)
THRESHOLD = 1.8

CELL_SIZE = 4
LBP_POINTS = 24
LBP_RADIUS = 3
#CELL_SIZE = 8
#LBP_POINTS = 8
#LBP_RADIUS = 2

svm = pickle.load(open(sys.argv[1]))
target = color.rgb2gray(io.imread(sys.argv[2]))
target_scaled = target + 0

sizexy = max(target_scaled.shape[0], target_scaled.shape[1])
print "imae size(y,x)", target_scaled.shape[0], target_scaled.shape[1]

scale_factor = 2.0 ** (-1.0 / 8.0)
detections = []


def overlap_score(a,b):
    left = max(a['x'], b['x'])
    right = min(a['x'] + a['width'], b['x'] + b['width'])
    top = max(a['y'], b['y'])
    bottom = min(a['y'] + a['height'], b['y'] + b['height'])
    intersect = max(0, (right - left) * (bottom - top))
    union = a['width'] * a['height'] + b['width'] * b['height'] - intersect
    return intersect / union

def get_histogram(image):
    print "image size", image.shape[0], image.shape[1]
    lbp = feature.local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, 'uniform')
    bins = LBP_POINTS + 2
    histogram = np.zeros(shape = (image.shape[0] / CELL_SIZE, image.shape[1] / CELL_SIZE, bins), dtype = np.int)
    
    for y in range(0, image.shape[0] - CELL_SIZE, CELL_SIZE):
        for x in range(0, image.shape[1] - CELL_SIZE, CELL_SIZE):
            for dy in range(CELL_SIZE):
                for dx in range(CELL_SIZE):
                    histogram[y / CELL_SIZE, x / CELL_SIZE, int(lbp[y + dy, x + dx])] += 1
    return histogram

for s in range(16):
    print "################# ", s
    histogram = get_histogram(target_scaled)
    print "histogram", histogram.shape[0], histogram.shape[1], histogram.shape[2]
    scale = (scale_factor ** s)
    print "scale ", scale

    for y in range(0, histogram.shape[0] - HEIGHT / CELL_SIZE):
        for x in range(0, histogram.shape[1] - WIDTH / CELL_SIZE):
            features = histogram[y:y + HEIGHT / CELL_SIZE,
                          x:x + WIDTH / CELL_SIZE].reshape(1, -1)

            score = svm.decision_function(features)

            if score[0] > THRESHOLD:
                scell = CELL_SIZE / scale
                swidth = WIDTH / scale
                sheight = HEIGHT / scale
                print "score,scale,y,x,whidth,height", round(score[0],3), round(scale,3), round(y*scell,0), round(x*scell,0), round(swidth,0), round(sheight,0)
                detections.append({
                    'x': x * CELL_SIZE / scale,
                    'y': y * CELL_SIZE / scale,
                    'width': WIDTH / scale,
                    'height': HEIGHT / scale,
                    'score': score[0]})
    target_scaled = transform.rescale(target_scaled, scale_factor)

print("-- detect -- ")

plt.title('detect result')
image = io.imread(sys.argv[2])
print(image.shape[0])
print(image.shape[1])
plt.axis([0, image.shape[1], image.shape[0], 0])
#io.imshow(image)
#io.show()
plt.imshow(image)
#plt.plot([x1,x2], [y1, y2])
#plt.plot( [5, 35], [20, 20], 'r', lw=5 )

detections = sorted(detections, key = lambda d: d['score'], reverse = True)
deleted = set()

for i in range(len(detections)):
    if i in deleted: continue
    for j in range(i +1, len(detections)):
        if overlap_score(detections[i], detections[j]) > 0.3:
            deleted.add(j)
detections = [d for i, d in enumerate(detections) if not i in deleted]


ar_detect = np.array(detections)
for idx in range(ar_detect.shape[0]):
    #print idx, ar_detect[idx]
    d_x1 = ar_detect[idx]['x']
    d_y1 = ar_detect[idx]['y']
    d_x2 = d_x1 + ar_detect[idx]['width']
    d_y2 = d_y1 + ar_detect[idx]['height']
    print "size ", round(ar_detect[idx]['width'],1), round(ar_detect[idx]['height'],1)
    plt.plot( [d_x1, d_x1], [d_y1, d_y2], 'r', lw=1 )
    plt.plot( [d_x1, d_x2], [d_y2, d_y2], 'r', lw=1 )
    plt.plot( [d_x2, d_x2], [d_y2, d_y1], 'r', lw=1 )
    plt.plot( [d_x2, d_x1], [d_y1, d_y1], 'r', lw=1 )


plt.show()

