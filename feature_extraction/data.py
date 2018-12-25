import pandas as pd


data_dir = "C:/Users/S/travel_data/dataset/"


def import_data():
    group_table = pd.read_csv(data_dir + "group.csv")
    print("import group_table")
    airline_table = pd.read_csv(data_dir + "airline_processed.csv")
    print("import airline_table")
    order_table = pd.read_csv(data_dir + "order.csv")
    print("import order_table")
    train_set = pd.read_csv("C:/Users/S/travel_data/training-set.csv")
    print("import train_set")
    test_set = pd.read_csv("C:/Users/S/travel_data/testing-set.csv")
    print("import test_set")
    return (group_table, airline_table, order_table, train_set, test_set)
    