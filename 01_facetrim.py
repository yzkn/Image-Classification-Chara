# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

import cv2
import sys
import os
import glob
from PIL import Image
import time


# 画像データはこのディレクトリのサブディレクトリに格納
root_dir = './traindata'

datadir = 'C:/opencv-4.0.1-vc14_vc15/sources/data'
haarcascades = [
    # 'haarcascades/haarcascade_frontalcatface.xml',
    # 'haarcascades/haarcascade_frontalcatface_extended.xml',
    'haarcascades/haarcascade_frontalface_alt.xml',
    'haarcascades/haarcascade_frontalface_alt_tree.xml',
    'haarcascades/haarcascade_frontalface_alt2.xml',
    'haarcascades/haarcascade_frontalface_default.xml',
    'haarcascades/haarcascade_profileface.xml',
    
    'haarcascades_cuda/haarcascade_frontalface_alt.xml',
    'haarcascades_cuda/haarcascade_frontalface_alt_tree.xml',
    'haarcascades_cuda/haarcascade_frontalface_alt2.xml',
    'haarcascades_cuda/haarcascade_frontalface_default.xml',
    'haarcascades_cuda/haarcascade_profileface.xml',
    ]


image_size = 100
count = 1

def image_data(file, haarcascade):
    global count
    cascade_file = datadir + '/' + haarcascade
    print('{} {}'.format(file, cascade_file))
    
    image = cv2.imread(file)
    image_gs = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # グレースケール画像を取得
    # =============================================================================
    #     scaleFactor 各画像スケールにおける縮小量を表します
    #     minNeighbors 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
    #     minSize 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
    #     (http://opencv.jp/opencv-2.1/cpp/object_detection.html)
    # =============================================================================
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔領域を取得
    face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(100,100))
    if len(face_list)>0:
        for i,face in enumerate(face_list):
            x,y,w,h = face
            face = image[y:y+h,x:x+w]
            face = cv2.resize(face,(image_size,image_size))
            newdir = str(os.path.join(*file.split(os.sep)[0:-1]))+'_face'
            if not os.path.exists(newdir):
                os.mkdir(newdir)
            cv2.imwrite(newdir+'/'+str(count)+'.png',face)
            count += 1
            print('    {} {}'.format(file, count))
    else:
        print('no face')

def main():
    subdirs = glob.glob(root_dir+'/**')
    for subdir in subdirs:
        count = 1
        print('sub directory: {}'.format(subdir))
        files = glob.glob(subdir+'/*.jpg')
        c=0
        for file in files:
            for haarcascade in haarcascades:
                time.sleep(0.1)
                print('  {:.2%} {}'.format((c/len(files)), file))
                image_data(file.replace('/', os.sep), haarcascade)
                c+=1


if __name__ == '__main__':
    main()