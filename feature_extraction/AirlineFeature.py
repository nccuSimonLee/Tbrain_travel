# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 旅行團會經過的機場數量
def airport_process(tmp_table):
    data = pd.DataFrame(columns=["group_id", "airport_amount", "abroad_airport",
                                 "abroad_time", "home_airport", "home_time"])
    gpid = tmp_table["group_id"].iloc[0]
    amount = len(set(tmp_table["src_airport"]).union(tmp_table["dst_airport"]))
    data["group_id"] = [gpid]
    data["airport_amount"] = [amount]
    
    data["abroad_airport"] = [tmp_table["src_airport"].iloc[0]]
    data["abroad_time"] = [tmp_table["fly_time"].iloc[0]]
    data["home_airport"] = [tmp_table["dst_airport"].iloc[-1]]
    data["home_time"] = [tmp_table["arrive_time"].iloc[-1]]
    return data

# 出國時的機場與回國時的機場是否一致
def airport_consistent(tmp_table):
    if tmp_table["src_airport"].iloc[0] == tmp_table["dst_airport"].iloc[-1]:
        return 1
    else:
        return 0

def get_part_of_day(hour):
    return (
        0 if 6 <= hour <= 11
        else
        1 if 12 <= hour <= 17
        else
        2 if 18 <= hour <= 23
        else
        3
    )

def airline_feature(airline_table, drop_feature):
    table = airline_table.copy()
    table["fly_time"] = pd.to_datetime(table["fly_time"])
    table["arrive_time"] = pd.to_datetime(table["arrive_time"])
    table["time_per_dist"] = table["flight_time"] / table["flight_dist"]
    
    gb = table.groupby("group_id", as_index=False)
    new_table = gb.agg({"flight_time": {"flight_time_sum": "sum",
                                        "flight_time_mean": "mean",
                                        "flight_time_std": "std",
                                        "flight_time_min": "min",
                                        "flight_time_max": "max"},
                        "flight_dist": {"flight_dist_sum": "sum",
                                        "flight_dist_mean": "mean",
                                        "flight_dist_std": "std",
                                        "flight_dist_min": "min",
                                        "flight_dist_max": "max"},
                        "time_per_dist": {"time_dist_sum": "sum",
                                          "time_dist": "mean",
                                          "time_dist_std": "std",
                                          "time_dist_min": "min",
                                          "time_dist_max": "max"}})
    new_colname = [new_table.columns.values[0][0]]
    for col in new_table.columns.values[1:]:
        new_colname.append(col[1])
    new_table.columns = new_colname
    
    new_table2 = gb.apply(airport_process)
    abroad_time = new_table2["abroad_time"]
    new_table2["abroad_hour"] = abroad_time.dt.hour.astype(int)
    new_table2["abroad_part_of_day"] = list(map(lambda abroad_hour: get_part_of_day(abroad_hour),
                                               abroad_time.dt.hour))
    new_table2["abroad_part_of_day"] = new_table2["abroad_part_of_day"].astype(int)
    #new_table["abroad_year"] = abroad_time.dt.year
    new_table2["abroad_DoY"] = abroad_time.dt.dayofyear.astype(int)
    new_table2["abroad_DoW"] = abroad_time.dt.dayofweek.astype(int)
    new_table2["abroad_DoM"] = abroad_time.dt.day.astype(int)

    home_time = new_table2["home_time"]
    #new_table["home_year"] = home_time.dt.year
    new_table2["home_DoW"] = home_time.dt.dayofweek.astype(int).astype(int)
    new_table2["home_DoM"] = home_time.dt.day.astype(int)
    new_table2["home_DoY"] = home_time.dt.dayofyear.astype(int)
    new_table2["home_hour"] = home_time.dt.hour.astype(int)
    new_table2["home_part_of_day"] = list(map(lambda home_hour: get_part_of_day(home_hour),
                                               home_time.dt.hour))
    new_table2["home_part_of_day"] = new_table2["home_part_of_day"].astype(int)
    
    airports = list(set(new_table2.abroad_airport).union(set(new_table2.home_airport)))
    airports_map = {}
    for i, ap in enumerate(airports):
        airports_map[ap] = i
    for col in ["abroad_airport", "home_airport"]:
        new_table2[col] = new_table2[col].map(airports_map)

    new_table = pd.merge(new_table, new_table2, on="group_id", how="left")
    drop_feature.extend(["abroad_time", "home_time"])
    # new_table.drop(columns=["abroad_time", "home_time"], inplace=True)
    del table
    return new_table
