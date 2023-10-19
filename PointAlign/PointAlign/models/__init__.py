# -*- coding: utf-8 -*-
# @Time    : 2023-05-14
# @Author  : lab
# @desc    :
# ----------------------------------------------------------------------------------------------------------------------
# importlib.import_module调用模型的时候，自动调用
# ----------------------------------------------------------------------------------------------------------------------
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'PointNet'))
