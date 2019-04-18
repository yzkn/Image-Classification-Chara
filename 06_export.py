# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

# remove the restriction of path strength before this pip command (LongPathsEnabled in the regedit)
# $ pip install tensorflowjs

from datetime import datetime
from glob import glob
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import datetime
import json
import logutil
import numpy as np
import os
import shutil
import sys
import tensorflowjs
import time

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)


# このスクリプトと同じディレクトリにdatasetフォルダを作成、
# そのサブディレクトリに訓練データと検証データからなるデータセットを格納
root_dirname = 'dataset'

root_train_dirname = 'train'
root_validate_dirname = 'validate'

# 学習時に重みを保存したディレクトリ
root_weight_dirname = 'weight'

# エクスポート先ディレクトリ
root_export_dirname = 'tfjs'

# テスト結果を出力するテキストファイル名
nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
result_filename = 'result-06_test-'+nowstr+'.txt'
result_json_filename = 'class.json'

# リサイズ後のサイズ
image_size = 100


def model_load(nb_classes):
    # VGG16をベースに
    # FC層(VGGの、1000クラスの分類結果を判別する層)は使用せずに、nb_classesクラスの判別を行うのでinclude_top=False
    input_tensor = Input(shape=(image_size, image_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    # FC層を新規作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # vgg16のFC層以外に、top_modelを結合
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(scrpath, root_dirname,
                                    root_weight_dirname, 'finetuning.h5'))

    # 損失関数を指定(カテゴリカル)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(
        lr=1e-3, momentum=0.9), metrics=['accuracy'])

    return model


def export(classes):
    # モデルのロード
    model = model_load(len(classes))

    # モデルを変換
    tensorflowjs.converters.save_keras_model(model, os.path.join(
        scrpath, root_dirname, root_export_dirname))

    with open(os.path.join(scrpath, root_dirname, root_export_dirname, result_json_filename), mode='a') as f:
        json.dump(classes, f)

    mes = 'Complete. Classes: [' + ','.join(classes) + ']'
    print(mes)
    with open(os.path.join(scrpath, root_dirname, root_export_dirname, result_filename), mode='a') as f:
        f.write(mes)


def main():
    if os.path.exists(os.path.join(scrpath, root_dirname, root_weight_dirname)):
        subdirs = glob(os.path.join(
            scrpath, root_dirname, root_train_dirname, '**'))
        classes = []
        sum_traindata = 0
        for subdir in subdirs:
            if os.path.isdir(subdir):
                print('sub directory: {}'.format(subdir))
                num_traindata = len(glob(os.path.join(subdir, '**')))
                sum_traindata += num_traindata
                classes.append(os.path.basename(subdir))
        export(classes)


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception()
    finally:
        logutil.log_end()
