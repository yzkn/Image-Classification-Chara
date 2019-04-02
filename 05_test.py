# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil
import sys
import time

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdatasetフォルダを作成、
# そのサブディレクトリに訓練データと検証データからなるデータセットを格納
root_dirname = 'dataset_sample'

root_train_dirname = 'train'
root_validate_dirname = 'validate'

# 学習時に重みを保存したディレクトリ
root_weight_dirname = 'weight'

# テストデータを格納したディレクトリ
root_test_dirname = 'test'

# テスト結果を出力するテキストファイル名
result_filename = 'test.txt'

# リサイズ後のサイズ
image_size = 100


def model_load(nb_classes):
    # VGG16をベースに
    # FC層(VGGの、1000クラスの分類結果を判別する層)は使用せずに、nb_classesクラスの判別を行うのでinclude_top=False
    input_tensor = Input(shape=(image_size, image_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層を新規作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # vgg16のFC層以外に、top_modelを結合
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(scrpath, root_dirname, root_weight_dirname, 'finetuning.h5'))

    # 損失関数を指定(カテゴリカル)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])

    return model



def test(classes):
    # モデルのロード
    model = model_load(len(classes))

    with open(os.path.join(scrpath, root_dirname, root_test_dirname, result_filename), mode='a') as f:
        # テスト用画像取得
        testdatas = glob(os.path.join(scrpath, root_dirname, root_test_dirname, '*.png'))
        for testdata in testdatas:
            img = image.load_img(testdata, target_size=(image_size, image_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            # rescaleと同じ比率
            x = x / 255
            pred = model.predict(x)[0]

            # TODO:予測結果を出力しつつ、ラベルを基にファイルをコピー(目視確認用)
            top = 1
            top_indices = pred.argsort()[-top:][::-1]
            result = [(classes[i], pred[i]) for i in top_indices]
            print('{}: {}'.format(testdata, result))
            f.write('{}: {}\n'.format(testdata, result))
            shutil.copyfile(testdata, os.path.join(TEMP, os.path.basename(file)))
        f.write('Complete\n')


def main():
    if os.path.exists(os.path.join(scrpath, root_dirname, root_weight_dirname)):
        subdirs = glob(os.path.join(scrpath, root_dirname, root_train_dirname, '**'))
        classes = []
        sum_traindata = 0
        for subdir in subdirs:
            if os.path.isdir(subdir):
                print('sub directory: {}'.format(subdir))
                num_traindata = len(glob(os.path.join(subdir, '**')))
                sum_traindata += num_traindata
                classes.append(os.path.basename(subdir))
        test(classes)


if __name__ == '__main__':
    main()
