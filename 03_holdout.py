# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
import os

scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdataフォルダを作成、
# そのサブディレクトリ(*_face_expand)に画像データを格納
root_dirname = 'data'

# 訓練データと検証データに分割したデータセットをこのディレクトリに出力
root_dirname = 'dataset'


def main():
    subdirs = glob(os.path.join(scrpath, root_dirname, '*_face_expand'))
    # for subdir in subdirs:
    #     print('sub directory: {}'.format(subdir))
    #     files = glob(os.path.join(subdir, '*.png'))
    #     count_originalfile = 0
    #     for file in files:
    #         print('  {:.2%} {}'.format((count_originalfile/len(files)), file))
    #         newdir = str((os.path.sep).join(
    #             file.split(os.path.sep)[0:-1]))+'_expand'
    #         if not os.path.exists(newdir):
    #             os.mkdir(newdir)
    #         count_originalfile += 1


if __name__ == '__main__':
    main()
