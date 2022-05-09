import argparse
import os
import pandas as pd

from python.logging import log_params


def write_file(data, location):
    print("Writing to file...")
    print(location)

    # Combine all the output types somehow?
    with open(location, "w") as f:
        f.write(data)


def read_file(file):
    print("Reading from file...")
    data = pd.read_csv(file)

    return data


def parse_args(cmd_args):
    print("Parsing...")

    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", help="verbose output", action="store_true", default=False)
    parser.add_argument("--input_dir", help="directory containing the input data files", type=str,
                        default=os.path.join("data", "labeled"))
    parser.add_argument("--output_dir", help="directory to put the output in", type=str,
                        default=os.path.join("data", "preprocessed"))
    parser.add_argument("--pipe", help="run the raw data through the labeling pipeline", action="store_true",
                        default=False)
    parser.add_argument("--prep", help="preprocess the data", action="store_true", default=False)
    parser.add_argument("--run", help="create models from preprocessed data", action="store_true", default=False)
    parser.add_argument("--pattern", help="the pattern to use to choose the files", type=str, default="*imu.csv")
    parser.add_argument("--window_size", help="number of rows to look at at once", type=int, default=100)
    parser.add_argument("--sampling_interval", help="sampling interval for the window", type=int, default=50)
    parser.add_argument("--gather_results",
                        help="gathers results from --completed_results_dir and writes a summary in that same directory",
                        action="store_true", default=False)
    parser.add_argument("--completed_results_dir",
                        help="The root directory where ML results are found used with --gather_results",
                        default=os.path.join("results"))
    parser.add_argument("--jumps_only", help="only train on jumps", action="store_true", default=False)
    parser.add_argument("--waist", help="use the waist sensor", action="store_true", default=False)
    parser.add_argument("--right", help="use the right ankle sensor", action="store_true", default=False)
    parser.add_argument("--left", help="use the left ankle sensor", action="store_true", default=False)
    parser.add_argument("--main_axes", help="only use the three main axes (acc, gyro, mag)", action="store_true",
                        default=False)
    parser.add_argument("--full", help="label full movement", action="store_true", default=False)
    parser.add_argument("--get_feature_imp", help="get classifier feature importances", action="store_true", default=False)

    return parser.parse_args(cmd_args)
