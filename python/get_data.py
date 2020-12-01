import glob
import os
import sys


def get_data(args):
    print("Fetching data...")

    data_files = list(glob.iglob(os.path.join("../{}/*/{}".format(args.input_dir, args.pattern))))

    # does this replace it permanently, and would that be a problem?
    pattern = args.pattern.replace("imu", "info")

    info_files = list(glob.iglob(os.path.join("../{}/*/{}".format(args.input_dir, pattern))))

    # check order

    data = [data_files, info_files]

    return data
