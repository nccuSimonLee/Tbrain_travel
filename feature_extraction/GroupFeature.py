# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.models.doc2vec import Doc2Vec
import jieba
jieba.set_dictionary('./feature_extraction/dict.txt.big')

def group_feature(group_table, drop_feature):
    table = group_table.copy()
    table["day_price"] = table["price"] / table["days"]

    table["subline_area"] = table["sub_line"] + "_" + table["area"]
    le = LabelEncoder()
    table["subline_area"] = le.fit_transform(table["subline_area"])
    table["sub_line"] = table["sub_line"].str.split("_", expand=True)[2].astype(int)
    table["area"] = table["area"].str.split("_", expand=True)[2].astype(int)
    
    
    table["begin_date"] = pd.to_datetime(table["begin_date"])
    table["begin_month"] = table["begin_date"].dt.month
    table["begin_DoY"] = table["begin_date"].dt.dayofyear
    table["begin_DoW"] = table["begin_date"].dt.dayofweek
    table["begin_quarter"] = table["begin_date"].dt.quarter
 
    model= Doc2Vec.load("./feature_extraction/d2v.model")
    table.product_name.fillna("NaN", inplace=True)
    prod_map = {}
    corpus = list(table.product_name.unique())
    for i, cp in enumerate(corpus):
        prod_map[cp] = i
    table["product_name"] = table.product_name.map(prod_map)
    prod_vec = pd.DataFrame(columns=["product_name"] + ["prod_vec_" + str(i + 1) for i in range(20)],
                                index=range(len(corpus)))
    for i in range(len(corpus)):
        prod_vec.iloc[i] = [i] + list(model.docvecs[i])
    table = pd.merge(table, prod_vec, on="product_name", how="left")
   

    drop_feature.extend(["begin_date", "product_name", "promotion_prog"])
    # table.drop(columns=["begin_date", "product_name", "promotion_prog"],
    #           inplace=True)
    return table
