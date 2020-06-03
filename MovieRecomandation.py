#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:03:01 2020

@author: oussama
"""

import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore')

df  = pd.read_csv('filecsv/u.data', sep='\t', names=
                  ['user_id','item_id','rating','timestamp'])