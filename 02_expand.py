# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

# Required:
#   $ pip install keras_preprocessing tensorflow

from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
# import time

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdataフォルダを作成、
# そのサブディレクトリ(*_face)に画像データを格納
root_dirname = 'data'

# 1枚の入力画像に対して何枚の画像を出力するか
expand_num = 100


def expand_data(file, count_originalfile, output_dir):
    img = load_img(file)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    datagen = ImageDataGenerator(
        channel_shift_range=100.0,
        height_shift_range=0.1,
        horizontal_flip=0.1,
        rotation_range=360.0,
        shear_range=0.1,
        vertical_flip=0.1,
        width_shift_range=0.1,
        zoom_range=0.1,
    )
    g = datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=os.path.splitext(
        os.path.basename(file))[0], save_format='png')
    count_batch = 0
    for i in range(expand_num):
        print('       {:.2%} {} {}'.format((count_batch/expand_num),
                                      (os.path.splitext(os.path.basename(file))[0]), count_batch))
        batch = g.next()
        count_batch += 1


def main():
    subdirs = glob(os.path.join(scrpath, root_dirname, '*_face'))
    for subdir in subdirs:
        print('sub directory: {}'.format(subdir))
        files = glob(os.path.join(subdir, '*.png'))
        count_originalfile = 0
        for file in files:
            # time.sleep(0.1)
            print('  {:.2%} {}'.format((count_originalfile/len(files)), file))
            newdir = str((os.path.sep).join(
                file.split(os.path.sep)[0:-1]))+'_expand'
            if not os.path.exists(newdir):
                os.mkdir(newdir)
            expand_data(file, count_originalfile, newdir)
            count_originalfile += 1


if __name__ == '__main__':
    main()
