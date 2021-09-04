import random

import pandas as pd
import numpy as np
import os

from python.constants import JUMP_TYPES, INITIALS
from python.orion import augment_with_window_size
from itertools import combinations
from python.get_data import get_data
from python.utils import write_file, read_file


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def preprocess(args):
    print("Preprocessing...")

    data_files, _ = get_data(args)
    # print(data_files)

    dfs = []

    for data in data_files:
        # if data[-12] == "J":
        #     print("skipping J")
        #     continue

        initials = data[-12:-10]

        df = read_file(data)
        if args.main_axes:
            df = df.drop(df.columns[37:42], axis=1)
            df = df.drop(df.columns[23:28], axis=1)
            df = df.drop(df.columns[9:14], axis=1)
            cols = 9
        else:
            cols = 14

        if not args.left:
            df = df.drop(df.columns[2 * cols:-1], axis=1)  # W+R
            if not args.right:
                df = df.drop(df.columns[cols:2 * cols], axis=1)  # W
                if not args.waist:
                    raise Exception("no columns selected")
            elif not args.waist:
                df = df.drop(df.columns[:cols], axis=1)  # R
        elif not args.right:
            df = df.drop(df.columns[cols:2 * cols], axis=1)  # W+L
            if not args.waist:
                df = df.drop(df.columns[:cols], axis=1)  # L
        elif not args.waist:
            df = df.drop(df.columns[:cols], axis=1)  # R+L
        else:
            pass  # All

        target = df.jump
        df.drop(columns=["jump"], axis=1, inplace=True)

        augmented, _ = augment_with_window_size(df, args.window_size, args.sampling_interval, True, False, "max")

        augmented["jump"] = target

        augmented["player"] = INITIALS[initials]

        # if data[-12] == "M":
        #     augmented.to_csv(os.path.join(args.output_dir, "augmented", data[-12:-1]), index=False, header=True)

        dfs.append(augmented)

    final_df = pd.concat(dfs)
    final_df.to_csv(os.path.join(args.output_dir, "Preprocessed_{}_{}.csv".format(
        args.window_size, args.sampling_interval)), index=False, header=True)

    # # TODO: change if needed if number of players changes
    # indexes = range(11)
    #
    # files = list(chunks(data_files, 7))
    #
    # all_combos = [[1, 2, 3, 5, 6, 7, 9], [0, 4, 8, 10]]#list(combinations(indexes, 9))
    #
    # combos_to_use = [0, 1]#random.sample(range(len(all_combos)), 11)
    #
    # index = 0
    #
    # for combo in combos_to_use:
    #     file_indexes = all_combos[combo]
    #     files_to_use = []
    #
    #     for i in file_indexes:
    #         files_to_use.extend(files[i])
    #
    #     dfs = []
    #
    #     for file in files_to_use:
    #         df = read_file(file)
    #
    #         _, width = df.shape
    #
    #         if width != 43:
    #             print("wrong shaped file")
    #
    #         if df.isnull().values.any():
    #             print("Null values found")
    #
    #         if args.main_axes:
    #             df = df.drop(df.columns[37:42], axis=1)
    #             df = df.drop(df.columns[23:28], axis=1)
    #             df = df.drop(df.columns[9:14], axis=1)
    #             cols = 9
    #         else:
    #             cols = 14
    #
    #         if not args.left:
    #             df = df.drop(df.columns[2 * cols:-1], axis=1) # W+R
    #             if not args.right:
    #                 df = df.drop(df.columns[cols:2 * cols], axis=1) # W
    #                 if not args.waist:
    #                     raise Exception("no columns selected")
    #             elif not args.waist:
    #                 df = df.drop(df.columns[:cols], axis=1) # R
    #         elif not args.right:
    #             df = df.drop(df.columns[cols:2 * cols], axis=1) # W+L
    #             if not args.waist:
    #                 df = df.drop(df.columns[:cols], axis=1) # L
    #         elif not args.waist:
    #             df = df.drop(df.columns[:cols], axis=1) # R+L
    #         else:
    #             pass # All
    #
    #         df = augment_df(df, args.window_size, args.sampling_interval)
    #
    #         if df.isnull().values.any():
    #             print("Null values found")
    #
    #         dfs.append(df)
    #
    #     final_df = pd.concat(dfs)
    #
    #     if final_df.isnull().values.any():
    #         print("Null values found")
    #
    #     preds = final_df.iloc[:, -1]
    #
    #     filename = "Preprocessed_{}.csv".format(index)
    #     index += 1
    #
    #     final_df.to_csv(os.path.join(args.output_dir, filename), index=False, header=False)


def augment_df(df, window_size, overlap):
    # Create dataframe with window_size * num_columns + 1 columns
    augmented_df = pd.DataFrame(index=range((len(df.index) - window_size) // overlap + 1),
                                columns=range((len(df.columns) - 1) * window_size + 1))

    # print(augmented_df)

    j = 0

    for i in range(0, len(df.index) - window_size + 1, overlap):
        window = df.iloc[i:i + window_size]

        values = window.jump.values
        row = window.iloc[:, :-1].values.reshape(-1)

        if values[0] == 0 and values[-1] == 0 and 1 in values:
            row = np.concatenate((row, [1]))
        elif values[0] == 0 and values[-1] == 0 and 2 in values:
            row = np.concatenate((row, [2]))
        elif values[0] == 0 and values[-1] == 0 and 3 in values:
            row = np.concatenate((row, [3]))
        elif values[0] == 0 and values[-1] == 0 and 4 in values:
            row = np.concatenate((row, [4]))
        elif values[0] == 0 and values[-1] == 0 and 5 in values:
            row = np.concatenate((row, [5]))
        elif values[0] == 0 and values[-1] == 0 and 6 in values:
            row = np.concatenate((row, [6]))
        elif values[0] == 0 and values[-1] == 0 and 7 in values:
            row = np.concatenate((row, [7]))
        elif values[0] == 0 and values[-1] == 0 and 8 in values:
            row = np.concatenate((row, [8]))
        else:
            row = np.concatenate((row, [0]))

        augmented_df.iloc[j] = row
        j += 1

    # print(augmented_df)
    print("Done augmenting!")

    return augmented_df


def label_df(df, info, full):

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

