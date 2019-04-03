# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
import os
import random
import shutil

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにresult01フォルダ or result02フォルダ、
# そのサブディレクトリに画像データが格納されている
#input_dirname = 'result01'
input_dirname = 'result02'

# 訓練データと検証データに分割したデータセットをこのディレクトリに出力
output_dirname = 'dataset'

output_train_dirname = 'train'    # 訓練データ
output_validate_dirname = 'validate'  # 検証データ

# 訓練データに振り分ける割合
ratio_train = 0.8


def main():
    subdirs = glob(os.path.join(scrpath, input_dirname, '**'))
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir))
            newdirs = [
                os.path.join(scrpath, output_dirname,
                             output_train_dirname, os.path.basename(subdir)),
                os.path.join(scrpath, output_dirname,
                             output_validate_dirname, os.path.basename(subdir))
            ]
            for newdir in newdirs:
                if not os.path.exists(newdir):
                    os.makedirs(newdir, exist_ok=True)

            files = glob(os.path.join(subdir, '*.png'))
            count_originalfile = 0
            for file in files:
                print('  {:.2%} {} {}'.format(
                    (count_originalfile/len(files)), subdir, file))
                if os.path.isfile(file):
                    if random.random() <= ratio_train:
                        shutil.copyfile(file, os.path.join(
                            newdirs[0], os.path.basename(file)))
                    else:
                        shutil.copyfile(file, os.path.join(
                            newdirs[1], os.path.basename(file)))
                count_originalfile += 1


if __name__ == '__main__':
    main()
