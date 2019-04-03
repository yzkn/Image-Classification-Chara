# Copyright (c) 2019 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.

import datetime
import os
import sys
import traceback


def log_start(file=__file__):
    try:
        scrpath = os.path.abspath(os.path.dirname(file))
        nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if not os.path.exists(os.path.join(scrpath, 'log')):
            os.makedirs(os.path.join(scrpath, 'log'), exist_ok=True)
        log_filepath = os.path.join(scrpath, 'log', os.path.splitext(
            os.path.basename(file))[0] + '_info_' + nowstr + '.txt')
        sys.stdout = open(log_filepath, 'a')
    except:
        log_exception()


def log_end():
    try:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    except:
        log_exception()



def log_exception(file=__file__):
    scrpath = os.path.abspath(os.path.dirname(file))
    nowstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(os.path.join(scrpath, 'log')):
        os.makedirs(os.path.join(scrpath, 'log'), exist_ok=True)
    log_filepath = os.path.join(scrpath, 'log', os.path.splitext(
        os.path.basename(file))[0] + '_error_' + nowstr + '.txt')

    t, v, tb = sys.exc_info()
    with open(log_filepath, 'a') as f:
        print(traceback.format_exception(t, v, tb), file=f)
        print(traceback.format_tb(e.__traceback__), file=f)
