#!/usr/bin/env python3

# SBATCH --time=48:00:00   # time limit
# SBATCH --nodes=1

# SBATCH --mem-per-cpu=131072M   # memory per CPU core
# SBATCH -J "run python script"   # job name
# SBATCH --mail-user=mikimj97@gmail.com   # email address
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL
# SBATCH --array=0-12  # how many tasks in the array
# SBATCH -c 1   # one CPU core per task


import os
import sys

sys.path.append(os.getcwd())

import glob
from python.run_models import run
from python.utils import parse_args

import copy


array_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
print("hello from job {}".format(array_id))

run_args_list = []
run_args = []
recurring_run_args = []

dirs = sorted(glob.glob(os.path.join("data", "preprocessed", "jump*")))


window_sizes = ["200", "220", "240", "260", "280", "300", "320", "340", "360", "380", "400", "420", "440"]
sampling_intervals = ["5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "90",
                      "110", "130", "150", "170", "190", "210", "230", "250", "270", "290", "310", "330", "350", "370", "390"]

window_size = window_sizes[array_id]

recurring_run_args.append("--run")
recurring_run_args.append("--jumps_only")
recurring_run_args.append("--window_size")
recurring_run_args.append(window_size)
recurring_run_args.append("--sampling_interval")
recurring_run_args.append("--input_dir")
recurring_run_args.append("--output_dir")

for input_dir in dirs:
    for sampling_interval in sampling_intervals:
        if int(window_size) < int(sampling_interval):
            continue

        run_args.extend(recurring_run_args)
        run_args.insert(-2, sampling_interval)
        run_args.insert(-1, input_dir)

        output_dir = os.path.join("results", input_dir.split(os.path.sep)[-1])
        run_args.append(output_dir)
        run_args_list.append(copy.copy(run_args))
        run_args.clear()

# print(run_args_list)

for arguments in run_args_list:
    print("about to run, args: {}".format(arguments))
    args = parse_args(arguments)
    run(args)
    print("done running")

print("\n\n\nTRAINING SCRIPT HAS FINISHED SUCCESSFULLY\n\n\n")
