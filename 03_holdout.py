# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
import datetime
import logutil
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

output_train_dirname = 'train'        # 訓練データ
output_validate_dirname = 'validate'  # 検証データ
output_test_dirname = 'test'          # テストデータ

# 訓練データ、検証データ、テストデータに振り分ける割合(和が1でなくてもよい)
ratio_train = 0.06
ratio_validate = 0.02
ratio_test = 0.02


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
            print('sub directory: {}'.format(subdir),
                  datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            newdirs = [
                os.path.join(scrpath, output_dirname,
                             output_train_dirname, os.path.basename(subdir)),
                os.path.join(scrpath, output_dirname,
                             output_validate_dirname, os.path.basename(subdir)),
                os.path.join(scrpath, output_dirname,
                             output_test_dirname, os.path.basename(subdir))
            ]
            for newdir in newdirs:
                if not os.path.exists(newdir):
                    os.makedirs(newdir, exist_ok=True)

            files = glob(os.path.join(subdir, '*.png'))
            count_originalfile = 0
            for file in files:
                rnd = random.random()
                print('  {:.2%} {:.2} {} {}'.format(
                    (count_originalfile/len(files)), rnd, subdir, file), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                if os.path.isfile(file):
                    if rnd <= ratio_train:
                        shutil.copyfile(file, os.path.join(
                            newdirs[0], os.path.basename(file)))
                    elif rnd <= ratio_train + ratio_validate:
                        shutil.copyfile(file, os.path.join(
                            newdirs[1], os.path.basename(file)))
                    elif rnd <= ratio_train + ratio_validate + ratio_test:
                        shutil.copyfile(file, os.path.join(
                            newdirs[2], os.path.basename(file)))
                    else:
                        pass
                count_originalfile += 1


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception(e)
    finally:
        logutil.log_end()
