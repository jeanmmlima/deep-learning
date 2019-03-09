#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:36:34 2019

@author: jeanmarioml
"""

import numpy as np

def MAbsError(obtained,expected):
    N = len(obtained)
    return np.sum(abs(expected-obtained))/N

def MSError(obtained, expected):
    N = len(obtained)
    return np.sum(np.power((expected-obtained),2))/N

def RMSError(obtained,expected):
    return np.sqrt(MSError(obtained,expected))

def percentError(obtained, expected):
    return (abs(obtained-expected)) * 100


#teste

ob = np.array([0.3, 0.02, 0.89, 0.32])
ex = np.array([1, 0, 1, 0])

MAbsError(ob,ex)
MSError(ob,ex)
RMSError(ob,ex)
percentError(ob,ex)