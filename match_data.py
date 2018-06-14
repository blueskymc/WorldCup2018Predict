#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a detail page module '

__author__ = 'Ma Cong'

import pandas as pd
import numpy as np

def get_score(line):
    m = line[4]
    v = line[5]
    if line[4]>line[5]:
        return 2
    elif line[4]<line[5]:
        return 0
    else:
        return 1



# df_match = pd.read_csv('data\\match_data.csv')
# df_name32 = pd.read_csv('data\\name32.csv')
# df_rank = pd.read_csv('data\\rank.csv')
# df_name32 = pd.merge(df_name32, df_rank, on='chinese_name')
# df_match = pd.concat([df_match, pd.DataFrame(columns=['mt', 'vt', 'mt_grad', 'vt_grad'])])
# cols = ['id', 'mainteam', 'visitteam', 'matchtime', 'm_score', 'v_score', 'mt', 'vt', 'mt_grad', 'vt_grad']
# df_match = df_match.ix[:, cols]
# df_match.loc[:, 6:8] = 0
# df_match.loc[:, 8:10] = 300  # 没进世界杯的球队积分设为500
# df_match['score'] = df_match.apply(lambda x: get_score(x), axis=1)
# for index, row in df_match.iterrows():
#     for index_name, row_name in df_name32.iterrows():
#         if row[1] == row_name[3]:
#             df_match.loc[index:index, 'mt'] = row_name[0]
#             df_match.loc[index:index, 'mt_grad'] = row_name[5]
#         elif row[2] == row_name[3]:
#             df_match.loc[index:index, 'vt'] = row_name[0]
#             df_match.loc[index:index, 'vt_grad'] = row_name[5]
#
# df_match.to_csv('new_match.csv', encoding='gbk')
# #df_name32.to_csv('new_name.csv', encoding='gbk')
# #print(df_match)
#
# df_wc2018 = pd.read_csv('data\\wc2018.csv')
# df_wc2018 = pd.concat([df_wc2018, pd.DataFrame(columns=['mt', 'vt', 'mt_grad', 'vt_grad'])])
# cols = ['home_team', 'visiting_team', 'mt', 'vt', 'mt_grad', 'vt_grad']
# df_wc2018 = df_wc2018.ix[:, cols]
# df_wc2018.loc[:, 2:4] = 0
# df_wc2018.loc[:, 4:] = 300  # 没进世界杯的球队积分设为500
# for index, row in df_wc2018.iterrows():
#     for index_name, row_name in df_name32.iterrows():
#         if row[0] == row_name[3]:
#             df_wc2018.loc[index:index, 'mt'] = row_name[0]
#             df_wc2018.loc[index:index, 'mt_grad'] = row_name[5]
#         elif row[1] == row_name[3]:
#             df_wc2018.loc[index:index, 'vt'] = row_name[0]
#             df_wc2018.loc[index:index, 'vt_grad'] = row_name[5]
#
# df_wc2018.to_csv('new_wc2018.csv', encoding='gbk')

def get_train_data():
    df = pd.read_csv('new_match.csv', encoding='gbk')
    df.drop(index=0, axis=0)
    x = np.array(df.ix[:, 7:11], dtype=np.float32)
    y = np.array(df.ix[:, 11:12], dtype=np.int)
    y = y[:, 0]
    index = np.arange(0, len(x), 1, dtype=np.int)
    np.random.seed(232)
    np.random.shuffle(index)
    index_train = index[int(len(x) * 0.1):]
    index_test = index[:int(len(x) * 0.1)]
    x_train = x[index_train]
    y_train = y[index_train]
    x_test = x[index_test]
    y_test = y[index_test]
    np.random.seed()
    return x_train.tolist(), y_train.tolist(), x_test.tolist(), y_test.tolist()

def get_predict_data():
    df = pd.read_csv('new_wc2018.csv', encoding='gbk')
    df.drop(index=0, axis=0)
    x = np.array(df.ix[:, 3:7], dtype=np.float32)
    return x.tolist()


