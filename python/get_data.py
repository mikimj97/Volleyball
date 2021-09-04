import glob
import os
import sys


def get_data(args):
    print("Fetching data...")

    path = os.path.join(args.input_dir, "*", args.pattern)

    data_files = list(glob.iglob(path))

    # # does this replace it permanently, and would that be a problem?
    # pattern = args.pattern.replace("imu", "info")
    path = path.replace("imu", "info")

    info_files = list(glob.iglob(path))

    return data_files, info_files
