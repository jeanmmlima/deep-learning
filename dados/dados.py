# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:37:41 2019

@author: Jean Mário
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
box_jenkins = pd.read_csv(
    '../Data/Online Prediction/Box-Jenkins_Gas-furnace.txt',
    header=None,
    names=['Input Gas Rate', 'CO2 (%)'])

# Visualize data
box_jenkins.head(2)