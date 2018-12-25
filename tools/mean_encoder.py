#coding: utf-8

import pandas as pd
from sklearn.model_selection import StratifiedKFold


# the target value column of input must be kept
def mean_encoder(input_table, y, columns, target, cv):
    input_table[target] = y
    table = input_table.copy()
    prior = table[target].mean()
    new_col = {}
    for col in columns:
        colname = col + "_mean_target"
        table[colname] = ""
        new_col[colname] = prior
    for tr_ind, val_ind in cv.split(y, y):
        x_tr, x_val = input_table.iloc[tr_ind], input_table.iloc[val_ind]
        for col in columns:
            colname = col + "_mean_target"
            means = x_val[col].map(x_tr.groupby(col)[target].mean())
            x_val[colname] = means
        table.iloc[val_ind] = x_val
    table.fillna(new_col, inplace=True)
    input_table.drop(columns=[target], inplace=True)
    return table

def test_set_encoder(train_set, train_y, test_set, columns, target):
    new_test = test_set.copy()
    train_set[target] = train_y
    prior = train_set[target].mean()
    new_col = {}
    for col in columns:
        colname = col + "_mean_target"
        means = new_test[col].map(train_set.groupby(col)[target].mean())
        new_test[colname] = means
        new_col[colname] = prior
    new_test.fillna(new_col, inplace=True)
    train_set.drop(columns=[target], inplace=True)
    return new_test

"""
cv = StratifiedKFold(5, shuffle=True, random_state=851206)
new_train_set = mean_encoder(train_set, ["source1_unit"], "deal_or_not", cv)
new_test = test_set_encoder(new_train_set, test_set, ["source1_unit"],"deal_or_not")
"""