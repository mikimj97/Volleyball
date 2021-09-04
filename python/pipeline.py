import os
import pandas as pd

from python.constants import JUMP_TYPES
from python.get_data import get_data


def run_pipeline(args):
    data_files, info_files = get_data(args)

    for file, info in zip(data_files, info_files):
        labeled_df = label_df(file, info, args.full)

        if labeled_df.isnull().values.any():
            print("Null values found")

        path = os.path.normpath(file).split(os.sep)
        path = os.path.join(args.output_dir, path[-2], path[-1])
        labeled_df.to_csv(path, index=False)


def label_df(file, info, full):
    print(file)
    df = pd.read_csv(file, header=0, index_col=0)

    headers = get_header_from_multi_header(df)
    df.columns = headers
    df = df.drop(df.index[0:2])
    df.reset_index(inplace=True, drop=True)
    df["jump"] = 0

    # TODO: What do we do about directions and different sensor locations?

    info_df = pd.read_csv(info)

    jumps = info_df.iloc[:, 0]
    start_times = info_df.iloc[:, 1]
    takeoff_times = info_df.iloc[:, 2]
    landing_times = info_df.iloc[:, 3]
    end_times = info_df.iloc[:, 4]
    print(jumps)
    print(start_times)
    print(takeoff_times)
    print(landing_times)
    print(end_times)

    jump_types = JUMP_TYPES["prep"]

    # TODO: do we need takeoff and landing for labeling? Probably, but how are they used?

    for start, takeoff, landing, end, jump in zip(start_times, takeoff_times, landing_times, end_times, jumps):
        # Label with jump type
        if start > takeoff:
            raise Exception("start is too big")
            exit(0)
        if takeoff > landing:
            raise Exception("takeoff is too big")
            exit(0)
        if landing > end:
            raise Exception("landing is too big")
            exit(0)

        print(jump)
        print(jump_types[jump])

        # Label everything from start of movement to end, or just from takeoff to landing
        if full:
            df.loc[start:end, "jump"] = jump_types[jump]
        else:
            df.loc[takeoff:landing, "jump"] = jump_types[jump]

    return df


def get_header_from_multi_header(df):
    """
    Gathers the names we'll actually use as headers aka "Accelerometer-X" instead of having a two layered column
    Args:
        df: a Pandas DataFrame of the IMU file
    Returns:
        a list of strings that will become the actual header values
    """
    # get the first three columns
    headers = df.iloc[0:2, ].to_numpy()
    # gather the real names for one run
    list_of_real_columns = []
    curr_device = ""

    for index, (device, axis) in enumerate(zip(headers[0], headers[1])):
        if not pd.isnull(device) and device.strip() != "":
            curr_device = device
        if curr_device == "Barometer" and pd.isnull(axis):
            axis = "pressure"
        list_of_real_columns.append(curr_device + "-" + axis)

    return list_of_real_columns

# def label_single_file(df, info):
#     labeled_df = pd.DataFrame()
#
#     return labeled_df
