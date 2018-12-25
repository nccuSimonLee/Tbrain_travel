# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def order_feature(order_table, drop_feature):
    table = order_table.copy()
    le = LabelEncoder()
    table["source1_source2"] = table["source_1"] + table["source_2"]
    table["source1_unit"] = table["source_1"] + table["unit"]
    table["source2_unit"] = table["source_2"] + table["unit"]
    table["source1_source2_unit"] = table["source_1"] + table["source_2"] + table["unit"]
    for col in ["source1_source2", "source1_unit", "source2_unit", "source1_source2_unit"]:
        table[col] = le.fit_transform(table[col])
    
    table["source_1"] = table["source_1"].str.split("_", expand=True)[2].astype(int)
    table["source_2"] = table["source_2"].str.split("_", expand=True)[2].astype(int)
    table["unit"] = table["unit"].str.split("_", expand=True)[2].astype(int)

    table["order_date"] = pd.to_datetime(table["order_date"])
    table["order_year"] = table["order_date"].dt.year
    table["order_month"] = table["order_date"].dt.month
    table["order_DoY"] = table["order_date"].dt.dayofyear
    table["order_quarter"] = table["order_date"].dt.quarter

    drop_feature.append("order_date")
    #table.drop(columns=["order_date"], inplace=True)
    return table
