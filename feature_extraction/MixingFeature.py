# coding:utf-8

import pandas as pd

def mixing_feature(input_table, drop_feature):
    table = input_table.copy()
    x = table["begin_date"]
    y = table["order_date"]
    table["order_begin_duration"] = list(map(lambda x_tmp: x_tmp.days, x - y))
    table["airport_days"] = table["airport_amount"] / table["days"]
    table["airport_price"] = table["airport_amount"] / table["price"]
    table["peopleamount_days"] = table["people_amount"] / table["days"]
    table["peopleamount_price"] = table["people_amount"] / table["price"]
    table["fltime_days"] = table["flight_time_sum"] / table["days"]
    table["fldist_days"] = table["flight_dist_sum"] / table["days"]
    table["fltime_price"] = table["flight_time_sum"] / table["price"]
    table["fldist_price"] = table["flight_dist_sum"] / table["price"]
    
    x = table.group_id.value_counts()
    group_id_map = {}
    for gpid, value in zip(x.index, x):
        group_id_map[gpid] = value
    table["group_id_count"] = table.group_id.map(group_id_map)
    
    accum = pd.read_csv("./feature_extraction/accum_order.csv")
    accum.order_id = accum.order_id.astype(str)
    table = pd.merge(table, accum, on="order_id", how="left")
    
    return table
