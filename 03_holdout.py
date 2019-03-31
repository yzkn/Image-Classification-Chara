# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
import os
import random
import shutil

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdataフォルダを作成、
# そのサブディレクトリ(*_face_expand)に画像データを格納
root_dirname = 'data'

# 訓練データと検証データに分割したデータセットをこのディレクトリに出力
output_dirname = 'dataset'

output_train_dirname = 'train'
output_validate_dirname = 'validate'


def main():
    subdirs = glob(os.path.join(scrpath, root_dirname, '*_face_expand'))
    for subdir in subdirs:
        print('sub directory: {}'.format(subdir))
        newdirs = [
            os.path.join(scrpath, output_dirname, output_train_dirname, os.path.basename(subdir)),
            os.path.join(scrpath, output_dirname, output_validate_dirname, os.path.basename(subdir))
        ]
        for newdir in newdirs:
            if not os.path.exists(newdir):
                os.makedirs(newdir, exist_ok=True)

        files = glob(os.path.join(subdir, '*.png'))
        count_originalfile = 0
        for file in files:
            print('  {:.2%} {} {}'.format((count_originalfile/len(files)), subdir, file))
            if os.path.isfile(file):
                if random.random() <= 0.8:
                    shutil.copyfile(file, os.path.join(newdirs[0], os.path.basename(file)))
                else:
                    shutil.copyfile(file, os.path.join(newdirs[1], os.path.basename(file)))
            count_originalfile += 1


if __name__ == '__main__':
    main()
