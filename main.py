import argparse
import sys
import os

from python.utils import parse_args
from python.preprocess import preprocess
from python.run_models import run
from python.pipeline import run_pipeline
from python.gather_results import gather_ml_results

if __name__ == "__main__":
    print(sys.argv)
    args = parse_args(sys.argv[1:])
    print(args)

    if args.pipe:
        print("Run pipeline")
        run_pipeline(args)
    elif args.prep:
        print("Preprocess")
        preprocess(args)
    elif args.run:
        print("Run")
        run(args)
    elif args.gather_results:
        print("Gather results")
        gather_ml_results(args)
    else:
        print("You must select one of these four options: '--run', '--pipe', '--prep', or  '--gather_results'.")


