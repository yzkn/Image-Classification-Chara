# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

# Required:
#   OpenCV (https://opencv.org/releases.html)
#   $ pip install numpy pillow opencv-python opencv-contrib-python

from glob import glob
from PIL import Image
import cv2
import logutil
import os
import time

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdataフォルダを作成、
# そのサブディレクトリに画像データが格納されている
input_dirname = 'data'

# 顔領域を切り出した画像をこのディレクトリに出力
output_dirname = 'result01'

# OpenCVのインストール先(README.md.txtがあるディレクトリ)
datadir = 'C:/opencv-4.0.1-vc14_vc15'

# リサイズ後のサイズ
image_size = 100
# 顔検出時の最小サイズ
min_face_size = 100

# 分類器のパス一覧
haarcascades = [
    # 'sources/data/haarcascades/haarcascade_frontalcatface.xml',
    # 'sources/data/haarcascades/haarcascade_frontalcatface_extended.xml',
    # 'sources/data/haarcascades/haarcascade_frontalface_alt.xml',
    # 'sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml',
    # 'sources/data/haarcascades/haarcascade_frontalface_alt2.xml',
    # 'sources/data/haarcascades/haarcascade_frontalface_default.xml',
    # 'sources/data/haarcascades/haarcascade_profileface.xml',

    # 'sources/data/haarcascades_cuda/haarcascade_frontalface_alt.xml',
    # 'sources/data/haarcascades_cuda/haarcascade_frontalface_alt_tree.xml',
    # 'sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml',
    # 'sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml',
    # 'sources/data/haarcascades_cuda/haarcascade_profileface.xml',
    'lbpcascade_animeface.xml',  # https://github.com/nagadomi/lbpcascade_animeface
]


def image_data(file, count_originalfile, haarcascade):
    cascade_file = os.path.join(datadir.replace('/', os.path.sep), haarcascade)

    image = cv2.imread(file)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # グレースケール画像を取得
    # =============================================================================
    #     scaleFactor 各画像スケールにおける縮小量を表します
    #     minNeighbors 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
    #     minSize 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
    #     (http://opencv.jp/opencv-2.1/cpp/object_detection.html)
    # =============================================================================
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔領域を取得
    face_list = cascade.detectMultiScale(
        image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(min_face_size, min_face_size))
    if len(face_list) > 0:
        count_face = 0
        for i, face in enumerate(face_list):
            x, y, w, h = face
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (image_size, image_size))
            newdir = os.path.join(
                scrpath,
                output_dirname,
                os.path.basename(
                    (os.path.sep).join(file.split(os.path.sep)[0:-1])
                )
            )
            if not os.path.exists(newdir):
                os.makedirs(newdir, exist_ok=True)
            output_file = os.path.join(newdir, str(
                count_originalfile)+'_'+str(count_face)+'.png')
            cv2.imwrite(output_file, face)
            count_face += 1
            print('         {:.2%} {}'.format(
                (count_face/len(face_list)), output_file))
    else:
        print('no face')


def main():
    subdirs = glob(os.path.join(scrpath, input_dirname, '**'))
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir))
            files = glob(os.path.join(subdir, '*.jpg'))
            count_originalfile = 0
            for file in files:
                for haarcascade in haarcascades:
                    time.sleep(0.1)
                    print('  {:.2%} {}'.format(
                        (count_originalfile/len(files)), file))
                    image_data(file, count_originalfile,
                               haarcascade.replace('/', os.path.sep))
                    count_originalfile += 1


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception()
    finally:
        logutil.log_end()
