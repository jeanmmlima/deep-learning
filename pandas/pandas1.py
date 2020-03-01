# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:56:42 2019

@author: Jean Mário
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#criando uma serie
series = pd.Series([1,3,5,np.nan,6,8])

#criando dataframe

dates = pd.date_range('20191101',periods=6)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

#describe() shows a quick statistic summary of your data:

df.describe()

#Transposing your data:

df.T

#Selection

#1. getting
df['A']
df[0:3]
df['20191101':'20191103']

#. By POSITION
 #position
df.iloc[0]
df.iloc[0:2]
 #slices
df.iloc[3:5,0:2]
#lista de posicoes
df.iloc[[1,2,4],[0,2]]

#slicing colunas e linhas de maneira explicita
df.iloc[1:3,:]
df.iloc[:,1:3]

#valor explicito
df.iloc[1,1]

#Boolean indexing
df[df.A > 0]


# Plots
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()

df = pd.DataFrame(np.random.randn(1000,4), index=ts.index, columns=['A','B','C','D'])
df = df.cumsum()

plt.figure()
df.plot()
plt.legend(loc='best')
plt.savefig('plotting-with-pandas.png')

# CSV files

df.to_csv("foo.csv")

df_tanks = pd.read_csv('data_tanks_quanser.csv')
#com tempo e indices 
tanks_t = pd.DataFrame(df_tanks.values, index=df_tanks.index, columns=['t','L2','V'])

#sem coluna do tempo
tanks = tanks_t.iloc[:,1:3]
tanks.plot()











