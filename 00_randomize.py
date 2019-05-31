# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

from glob import glob
import datetime
import logutil
import os
import random


scrpath = os.path.abspath(os.path.dirname(__file__))
os.chdir(scrpath)

# このスクリプトと同じディレクトリにdataフォルダ、
# そのサブディレクトリに画像データが格納されている
input_dirname = 'data'

digit_num = 10


def main():
    subdirs = glob(os.path.join(scrpath, input_dirname, '**'))
    for subdir in subdirs:
        if os.path.isdir(subdir):
            print('sub directory: {}'.format(subdir), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            files = glob(os.path.join(subdir, '*.png'))
            count_originalfile = 0
            for file in files:
                print('  {:.2%} {}'.format(
                    (count_originalfile/len(files)), file), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                rnd = random.randrange(10**digit_num, 10**(digit_num + 1))
                newfile = os.path.join(
                    # scrpath, input_dirname,
                    os.path.dirname(os.path.abspath(file)),
                    str(rnd) + '__' + os.path.basename(file)
                )
                i = 0
                while os.path.exists(newfile):
                    newfile = os.path.join(
                        scrpath,
                        input_dirname,
                        '{}'.format(rnd + i) + '__' + os.path.basename(file)
                    )
                else:
                    os.rename(file, newfile)
                count_originalfile += 1


if __name__ == '__main__':
    logutil.log_start(__file__)
    try:
        main()
    except Exception as e:
        logutil.log_exception(__file__, e)
    finally:
        logutil.log_end()
