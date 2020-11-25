import pandas as pd

from python.get_data import get_data
from python.utils import write_file


def preprocess(args):
    print("Preprocessing...")

    data = get_data(args)

    df = label_df(data)

    write_file(df)


def label_df(data):
    df = pd.DataFrame()

    for df, info in data:
        labeled_df = label_single_file(df, info)
        # append to main df

    return df


def label_single_file(df, info):
    labeled_df = pd.DataFrame()

    return labeled_df
