import argparse
import os

from python.logging import log_params


def write_file(file, location):
    print("Writing to file...")


def read_file(location):
    print("Reading from file...")


def parse_args(cmd_args):
    print("Parsing...")

    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", help="verbose output", action="store_true", default=False)
    parser.add_argument("--input_dir", help="directory containing the input data files", type=str,
                        default=os.path.join("data", "raw"))
    parser.add_argument("--output_dir", help="directory to put the output in", type=str,
                        default=os.path.join("data", "preprocessed"))
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

    return parser.parse_args(cmd_args)
