import numpy as np
import pandas as pd
import typing
from tqdm import tqdm


def augment_with_window_size(df: pd.DataFrame, window_size: int, window_sample_by: int,
                             subtract: bool = False, verbose: bool = False, aggregate: str = None,
                             filepath: str = None) -> typing.Tuple[typing.Any, np.ndarray]:
#    result_miki, desired_rows = augment_with_window_size(test_df, 150, 5, True, True, "max", None)

    """
    This is the main function to augment the data with it's previous timesteps, for the time series model

    Args:
        df: the DataFrame to be augmented
        window_size: an int indicating the amount of previous timesteps to augment the dataset with
        window_sampling_by: an int inidicating by what step are we augmenting the dataset by.  Aka with window_sampling_by == 5,
                            it samples every 5th previous frame
        subtract: whether to use the difference between the current frame and the previous frame or to use the original previous frame
        verbose: whether to use tqdm or not
        aggregate: whether to do mean, min, max pooling over the data

    Returns:
        The augmented DataFrame
    """
    # gather the columns that we will need to augment by
    columns_to_copy = [column_name for column_name in df.columns if "jump" not in column_name]
    if (((window_size // 2) % window_sample_by) == 0):
        number_of_additional_frames = window_size // window_sample_by
    else:
        number_of_additional_frames = (window_size // window_sample_by) + 2
    additional_columns = []
    # building the headers for additional_columns
    for (name, multiplier) in [("past", -1), ("future", 1)]:  # past and future frames
        for index in range(1,
                           number_of_additional_frames // 2 + 1):  # start at 1 for simplicity in reading to others
            additional_columns.extend(
                [column_name + "-" + name + "-" + str(index * window_sample_by) for column_name in columns_to_copy])

    # init with NaNs
    augment_df = pd.DataFrame(np.nan, columns=additional_columns, index=list(range(len(df))), dtype='float')

    # go through the previous frames to gather that data
    if verbose:
        # logger.info("Initializing tqdm to show augmentation row position...")
        loop = tqdm(total=df.shape[0], position=0, leave=True)
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    for (row_num, row) in df.iterrows():
        # create the rows to get (does not includes negative rows or rows past the max)
        padded_rows, desired_rows = gather_rows_by_sampling(row_num, row, df, window_sample_by, window_size, subtract,
                                              augment_df.shape[1], aggregate)
        augment_df.loc[row_num, :] = padded_rows
        if verbose:
            loop.update(1)

    # concat the augmented data with the original
    #np.savetxt(filepath.lower(), augment_df.values, fmt='%s', newline=' ')
    final_data = pd.concat([df, augment_df], axis=1)
    assert final_data.shape[0] == df.shape[0] == augment_df.shape[0], "row size did not stay constant, error"
    assert final_data.shape[1] == df.shape[1] + augment_df.shape[1], "column size did not add, error"
    return final_data, desired_rows


def gather_rows_by_sampling(row_num: int, row: pd.Series, df: pd.DataFrame, window_sample_by: int, window_size: int,
                            subtract: bool, aug_cols: int, aggregate: str) -> np.array:
    """
    The basic way of sampling, taking every `window_sample_by` row

    Args:
        row_num: the row number we're currently on
        row: the current row of data
        df: the main pandas DataFrame
        window_sample_by: how often to sample the data
        window_size: how big a range to sample out
        subtract: a bool indicating whether to use the difference or not
        aug_cols: the number of columns in the augmented DataFrame
        aggregate: None if no aggregate, or the type (i.e. mean, min, max, etc.)

    Returns:
        gathered row for the augmented DataFrame
    """
    row = row.astype(np.float32)
    if aggregate is None:
        rows_to_gather = generate_row_numbers(row_num, df.shape[0], window_sample_by, window_size)
        rows_gathered = df.iloc[rows_to_gather, :]
    else:
        rows_gathered = generate_aggregated_rows(row_num, df, window_sample_by, window_size, aggregate)

    if subtract:
        prev_row = row - rows_gathered
    rows_shaped = rows_gathered.to_numpy().reshape(-1)

    # find how many are left to fill, fill with nans
    if row_num - 0 < df.shape[0] - 1 - row_num:  # padding needs to be on whichever side is lacking
        padding = (aug_cols - rows_shaped.shape[0], 0)
    else:
        padding = (0, aug_cols - rows_shaped.shape[0])

    # print(padding)

    padded_rows = np.pad(rows_shaped.astype(np.float32), padding, "constant", constant_values=np.NaN)
    assert padded_rows.shape[0] == aug_cols, "shapes were not aligned for copying, padded: {} vs augment {}".format(
        padded_rows.shape[0], aug_cols)
    return padded_rows, rows_shaped


def generate_aggregated_rows(row_num, df, window_sample_by, window_size, aggregate: str) -> typing.List[int]:
    """
    Gathers the aggregated rows and returns them

    Args:
        row_num: current row number of iteration through the DataFrame
        df: the DataFrame for the IMU file
        window_sample_by: the rate to sample
        window_size: the window size to sample
        aggregate: the type of aggregation to do

    Returns:
        A list of indices indicating samples to gather
    """
    window_size_half = window_size // 2  # past and future
    # plus one to include range
    lower = df.iloc[row_num - window_size_half:row_num, :]
    upper = df.iloc[row_num + 1:row_num + window_size_half + 1, :]

    lower_list = []
    upper_list = []
    df_list = [lower, upper]
    for index, cur_list in enumerate([lower_list, upper_list]):
        count = len(df_list[index])
        counter = 0
        while count > 0:
            cur_list.extend([counter] * min(window_sample_by, len(df_list[index]) - len(cur_list)))
            counter += 1
            count -= window_sample_by

    try:
        lower["index"] = lower_list
        upper["index"] = upper_list
    except Exception as e:
        import pdb
        pdb.set_trace()
        print(e)

    # aggregate
    if aggregate == "mean":
        lower = lower.groupby("index").mean()
        upper = upper.groupby("index").mean()
    elif aggregate == "min":
        lower = lower.groupby("index").min()
        upper = upper.groupby("index").min()
    elif aggregate == "max":
        lower = lower.groupby("index").max()
        upper = upper.groupby("index").max()
    else:
        raise NotImplementedError()

    row_diffs = pd.concat([lower, upper], axis=0)  # combine the two
    return row_diffs


def generate_row_numbers(row_num, df_len, window_sample_by, window_size) -> typing.List[int]:
    """
    Generates the row numbers that should be added for the window
    I.E. given row 50 with window_sample_by = 2 and 4 windows, we gather [46, 48, 52, 54]

    Args:
        row_num: current row number of iteration through the DataFrame
        df_len: the max len of the DataFrame, to be used for as the maximum number
        window_sample_by: the rate to sample
        window_size: the window size to sample

    Returns:
        A list of indices indicating samples to gather
    """
    window_size_half = window_size // 2  # past and future
    # plus one to include range
    row_diffs_lower = ([row_num - window_sample_by * (index + 1) for index in
                        range(window_size_half // window_sample_by)])  # create the sampling
    row_diffs_lower.reverse()  # to reorder them
    row_diffs_upper = np.arange(row_num, row_num + window_size_half + 1, window_sample_by)  # create the sampling
    row_diffs = np.concatenate((row_diffs_lower, row_diffs_upper), axis=0)  # combine the two
    row_diffs = row_diffs[np.searchsorted(row_diffs, 0) :]  # make non-negative
    row_diffs = row_diffs[row_diffs <= df_len - 1].tolist()  # make smaller than max
    if row_num in row_diffs:
        row_diffs.remove(row_num)
    return row_diffs
