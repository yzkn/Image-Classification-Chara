# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from datetime import datetime
from glob import glob
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import datetime
import logutil
import numpy as np
import os
import shutil
import sys
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

# テストデータを格納したディレクトリ
root_test_dirname = 'test'

# テスト結果を出力するテキストファイル名
nowstr = datetime.now().strftime('%Y%m%d%H%M%S')
result_filename = 'result-05_test-'+nowstr+'.txt'

# テスト結果に応じてファイルを移動する先のディレクトリ
root_classified_dirname = 'classified_' + nowstr

# リサイズ後のサイズ
image_size = 100


# 正答率評価のための変数
count_items_all = 0
count_items_correct = 0


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


def test_files(classes):
    global count_items_all
    global count_items_correct

    # モデルのロード
    model = model_load(len(classes))

    # テスト用画像取得
    testdatas = glob(os.path.join(
        scrpath, root_dirname, root_test_dirname, '*.png'))
    for testdata in testdatas:
        test(model, classes, testdata, '')

    subdirs = glob(os.path.join(
        scrpath, root_dirname, root_test_dirname, '**'))
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir))
            testdatas = glob(os.path.join(subdir, '*.png'))
            count_testfile = 0
            for testdata in testdatas:
                print('  {:.2%} {}'.format(
                    (count_testfile/len(testdatas)), testdata))
                test(model, classes, testdata, os.path.basename(subdir))
                count_testfile += 1
    if count_items_all > 0:
        mes = 'Complete. accuracy:{} / {} = {:.2%}\n'.format(
            count_items_correct, count_items_all, count_items_correct / count_items_all)
    else:
        mes = 'Complete.'
    print(mes)
    with open(os.path.join(scrpath, root_dirname, root_classified_dirname, result_filename), mode='a') as f:
        f.write(mes)


def test(model, classes, testdata, correct_answer):
    global count_items_all
    global count_items_correct

    img = image.load_img(testdata, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # rescaleと同じ比率
    x = x / 255
    pred = model.predict(x)[0]

    top = 1
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    print('{}: {}'.format(testdata, result))
    with open(os.path.join(scrpath, root_dirname, root_classified_dirname, result_filename), mode='a') as f:
        f.write('{}: {}\n'.format(testdata, result))

    if correct_answer != '':
        count_items_all += 1
        if correct_answer == result[0][0]:
            count_items_correct += 1

    dstdir = os.path.join(
        scrpath, root_dirname, root_classified_dirname, correct_answer + '-' + result[0][0])
    if not os.path.exists(dstdir):
        os.makedirs(dstdir, exist_ok=True)
    shutil.copyfile(testdata, os.path.join(dstdir, os.path.basename(testdata)))


def main():
    if not os.path.exists(os.path.join(scrpath, root_dirname, root_classified_dirname)):
        os.mkdir(os.path.join(scrpath, root_dirname, root_classified_dirname))
    else:
        nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        os.rename(os.path.join(scrpath, root_dirname, root_classified_dirname), os.path.join(
            scrpath, root_dirname, root_classified_dirname + '_' + nowstr + '.bak'))
        os.mkdir(os.path.join(scrpath, root_dirname, root_classified_dirname))

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
        test_files(classes)


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception()
    finally:
        logutil.log_end()
