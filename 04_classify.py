# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import datetime
import logutil
import numpy as np
import os
import time

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdatasetフォルダを作成、
# そのサブディレクトリに訓練データと検証データからなるデータセットを格納
root_dirname = 'dataset'

root_train_dirname = 'train'
root_validate_dirname = 'validate'

# 結果出力先
root_weight_dirname = 'weight'

# リサイズ後のサイズ
image_size = 100

# パラメータ
batch_size = 16
nb_epoch = 10


def vgg_model_maker(nb_classes):
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
    # model = Model(input=vgg16.input, output=top_model(vgg16.output))
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    return model


def image_generator(classes):
    # rescale=1.0 / 255で、255レベルの画素値を0から1に正規化
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # dataset配下の画像ファイルを基にデータセット作成
    train_generator = train_datagen.flow_from_directory(
        os.path.join(scrpath, root_dirname, root_train_dirname),
        target_size=(image_size, image_size),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(scrpath, root_dirname, root_validate_dirname),
        target_size=(image_size, image_size),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    return (train_generator, validation_generator)


def train(classes, nb_train_samples, nb_validation_samples, steps=None):
    start = time.time()

    # steps_per_epoch_value = int(steps) if steps!=None else max(1, nb_train_samples//batch_size)
    # validation_steps_value = int(steps) if steps!=None else max(1, nb_validation_samples//batch_size)
    steps_per_epoch_value = int(nb_train_samples//batch_size) if steps==None else min(steps, nb_train_samples//batch_size)
    validation_steps_value = int(nb_validation_samples//batch_size) if steps==None else min(steps, nb_validation_samples//batch_size)

    print(
        'train( {}, {}, {} )'.format(classes, nb_train_samples, nb_validation_samples) +
        ' steps_per_epoch_value: {}; validation_steps_value: {}'.format(
            steps_per_epoch_value, validation_steps_value),
        datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    )

    vgg_model = vgg_model_maker(len(classes))

    # 最後の一層を除いて学習し直さない
    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    # 損失関数を指定(カテゴリカル)
    vgg_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(
        lr=1e-3, momentum=0.9), metrics=['accuracy'])
    train_generator, validation_generator = image_generator(classes)

    history = vgg_model.fit_generator(
        train_generator,
        validation_data=validation_generator,

        # nb_epoch=nb_epoch,
        # nb_val_samples=nb_validation_samples,
        # samples_per_epoch=nb_train_samples
        workers=1,
        use_multiprocessing=False,
        epochs=nb_epoch,
        steps_per_epoch=steps_per_epoch_value,
        validation_steps=validation_steps_value
    )

    steps_label = ("_" + str(steps)) if steps!=None else ""

    if not os.path.exists(os.path.join(scrpath, root_dirname, root_weight_dirname, steps_label)):
        os.makedirs(os.path.join(scrpath, root_dirname, root_weight_dirname, steps_label), exist_ok=True)
    vgg_model.save(os.path.join(
        scrpath, root_dirname, root_weight_dirname, steps_label, 'finetuning.h5'))

    process_time = (time.time() - start)
    print('Completed. {} sec'.format(process_time),
          datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))


def main():
    if not os.path.exists(os.path.join(scrpath, root_dirname, root_weight_dirname)):
        os.makedirs(os.path.join(scrpath, root_dirname,
                                 root_weight_dirname), exist_ok=True)
    else:
        nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        os.rename(os.path.join(scrpath, root_dirname, root_weight_dirname), os.path.join(
            scrpath, root_dirname, root_weight_dirname + '_' + nowstr + '.bak'))
        os.mkdir(os.path.join(scrpath, root_dirname, root_weight_dirname))

    subdirs = glob(os.path.join(
        scrpath, root_dirname, root_train_dirname, '**'))
    classes = []
    sum_traindata = 0
    sum_validate = 0
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir),
                  datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            if os.path.isdir(os.path.join(scrpath, root_dirname, root_validate_dirname, os.path.basename(subdir))):
                # print('  {:.2%} {} {}'.format((count_originalfile/len(files)), subdir, file), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                num_traindata = len(glob(os.path.join(subdir, '**')))
                num_validate = len(glob(os.path.join(
                    scrpath, root_dirname, root_validate_dirname, os.path.basename(subdir), '**')))
                sum_traindata += num_traindata
                sum_validate += num_validate
                classes.append(os.path.basename(subdir))

    # # step数100
    # train(classes, sum_traindata, sum_validate, 100)

    # step数を徐々に増加させる
    # max_steps = sum_traindata//batch_size
    # lst = [10**i for i in range(0,int(np.log10(max_steps))+1)]
    lst = [100, 200, 300]
    for l in lst:
        train(classes, sum_traindata, sum_validate, l)

    # # step数自動設定
    # train(classes, sum_traindata, sum_validate, None)


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception(__file__, e)
    finally:
        logutil.log_end()
