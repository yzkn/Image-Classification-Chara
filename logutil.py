# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

import datetime
import os
import sys
import traceback


def log_start(file=__file__):
    scrpath = ''
    try:
        # 引数からスクリプトが存在するディレクトリのパスを取得
        if os.path.exists(file) and os.path.isfile(file):
            scrpath = os.path.abspath(os.path.dirname(file))
        else:
            scrpath = os.getcwd()
        # 取得できなければカレントディレクトリを取得
        if not os.path.exists(scrpath):
            scrpath = os.getcwd()
        # サブフォルダlogを作成
        if not os.path.exists(os.path.join(scrpath, 'log')):
            os.makedirs(os.path.join(scrpath, 'log'), exist_ok=True)

        # ログファイルパスを生成
        if os.path.exists(os.path.join(scrpath, 'log')):
            nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            log_filepath = os.path.join(scrpath, 'log', os.path.splitext(
                os.path.basename(file))[0] + '_info_' + nowstr + '.txt')

            # 標準出力を切替
            sys.stdout = open(log_filepath, 'a')
    except:
        # 出力先を元に戻す
        sys.stdout.close()
        log_exception()


def log_end():
    try:
        # 出力先を元に戻す
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    except:
        log_exception()


def log_exception(file=__file__, e=None):
    scrpath = ''
    try:
        # 引数からスクリプトが存在するディレクトリのパスを取得
        if os.path.exists(file) and os.path.isfile(file):
            scrpath = os.path.abspath(os.path.dirname(file))
        else:
            scrpath = os.getcwd()
        # 取得できなければカレントディレクトリを取得
        if not os.path.exists(scrpath):
            scrpath = os.getcwd()
        # サブフォルダlogを作成
        if not os.path.exists(os.path.join(scrpath, 'log')):
            os.makedirs(os.path.join(scrpath, 'log'), exist_ok=True)

        # 例外ログファイルパスを生成
        if os.path.exists(os.path.join(scrpath, 'log')):
            nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            log_filepath = os.path.join(scrpath, 'log', os.path.splitext(
                os.path.basename(file))[0] + '_error_' + nowstr + '.txt')

            # 例外ログファイルに書き出し
            t, v, tb = sys.exc_info()
            with open(log_filepath, 'a') as f:
                print(traceback.format_exception(t, v, tb), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), file=f)
                print(traceback.format_tb(e.__traceback__), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), file=f)
    except:
        t, v, tb = sys.exc_info()
        nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        log_filepath = os.path.splitext(
            os.path.basename(file))[0] + '_error_' + nowstr + '.txt'
        with open(log_filepath, 'a') as f:
            print(traceback.format_exception(t, v, tb), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), file=sys.stderr)
            print(traceback.format_tb(e.__traceback__), datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), file=sys.stderr)
