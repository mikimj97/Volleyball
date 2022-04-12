#!/usr/bin/env python3

# SBATCH --time=96:00:00   # time limit
# SBATCH --nodes=1

# SBATCH --mem-per-cpu=131072M   # memory per CPU core
# SBATCH -J "run python script"   # job name
# SBATCH --mail-user=mikimj97@gmail.com   # email address
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL
# SBATCH --array=0-30  # how many tasks in the array
# SBATCH -c 1   # one CPU core per task


import os
import sys

sys.path.append(os.getcwd())

import glob
from python.run_models import run
from python.utils import parse_args
from python.preprocess import preprocess
import pandas as pd
import copy

array_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
print("hello from job {}".format(array_id))

prep_args_list = []
prep_args = []
recurring_prep_args = []

window_sizes = ["200", "220", "240", "260", "280", "300", "320", "340", "360", "380", "400", "420", "440"]
sampling_intervals = ["5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "90",
                      "110", "130", "150", "170", "190", "210", "230", "250", "270", "290", "310", "330", "350", "370", "390"]
# offsets = ["0.25", "0.275", "0.3", "0.325", "0.35"]
# post_offset = 0.3
# wavelet_scales = ["64", "74", "84", "94", "104", "114", "124"]


recurring_prep_args.append("--prep")
recurring_prep_args.append("--jumps_only")
recurring_prep_args.append("--pattern")
recurring_prep_args.append("*imu.csv")
recurring_prep_args.append("--main_axes")
recurring_prep_args.append("--waist")
recurring_prep_args.append("--left")
recurring_prep_args.append("--right")
recurring_prep_args.append("--input_dir")
recurring_prep_args.append("data/labeled/jump")
recurring_prep_args.append("--output_dir")
recurring_prep_args.append("data/preprocessed/jump_all_main")
# recurring_prep_args.append(sampling_interval)
# recurring_prep_args.append("--offset_before_takeoff")
# recurring_prep_args.append(1 - post_offset)
# recurring_prep_args.append("--offset_after_takeoff")
# recurring_prep_args.append(post_offset)

# These ones will get a value that changes
recurring_prep_args.append("--sampling_interval")
recurring_prep_args.append("--window_size")
# recurring_prep_args.append("--wavelet_scale")

sampling_interval = sampling_intervals[array_id]

for window_size in window_sizes:

    if int(window_size) < int(sampling_interval):
        continue

# for wavelet_scale in wavelet_scales:
    # for post_offset in offsets:
    #     pre_offset = str(1 - float(post_offset))

    prep_args.extend(recurring_prep_args)
    # prep_args.insert(-2, window_size)
    #     prep_args.insert(-4, sampling_interval)
    prep_args.insert(-1, sampling_interval)
    #     prep_args.insert(-2, pre_offset)
    #     prep_args.insert(-1, post_offset)
    prep_args.append(window_size)

    prep_args_list.append(copy.copy(prep_args))
    prep_args.clear()

# print(prep_args_list)


for arguments in prep_args_list:
    print("about to preprocess, args: {}".format(arguments))
    args = parse_args(arguments)
    preprocess(args)
    print("done preprocessing")


print("\n\n\nPREPROCESS SCRIPT HAS FINISHED SUCCESSFULLY\n\n\n")
