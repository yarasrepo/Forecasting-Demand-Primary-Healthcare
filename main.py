import pandas as pd

all_cols_train = pd.read_csv("./input/all_columns/df_train_collated.csv")

cols_to_drop = [
    "w_{t-1}", "w_{t-2}", "w_{t-3}", "w_{t-4}", "w_{t-5}",
    "w_{t-1}_trend", "w_{t-1}_seasonality"
]

all_cols_train.drop(columns=cols_to_drop, inplace=True)

