# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

# Required:
#   $ pip install keras_preprocessing tensorflow

from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
import datetime
import logutil
import matplotlib.pyplot as plt
import numpy as np
import os

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにresult01フォルダ、
# そのサブディレクトリに画像データが格納されている
input_dirname = 'result01'

# 水増しした画像をこのディレクトリに出力
output_dirname = 'result02'

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
    if not os.path.exists(os.path.join(scrpath, output_dirname)):
        os.mkdir(os.path.join(scrpath, output_dirname))
    else:
        nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        os.rename(os.path.join(scrpath, output_dirname), os.path.join(
            scrpath, output_dirname + '_' + nowstr + '.bak'))
        os.mkdir(os.path.join(scrpath, output_dirname))

    subdirs = glob(os.path.join(scrpath, input_dirname, '**'))
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir))
            files = glob(os.path.join(subdir, '*.png'))
            count_originalfile = 0
            for file in files:
                # time.sleep(0.1)
                print('  {:.2%} {}'.format(
                    (count_originalfile/len(files)), file))
                newdir = os.path.join(
                    scrpath,
                    output_dirname,
                    os.path.basename(
                        (os.path.sep).join(file.split(os.path.sep)[0:-1])
                    )
                )
                if not os.path.exists(newdir):
                    os.makedirs(newdir, exist_ok=True)
                expand_data(file, count_originalfile, newdir)
                count_originalfile += 1


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception()
    finally:
        logutil.log_end()
