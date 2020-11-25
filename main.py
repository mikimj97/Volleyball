import argparse
import sys
import os

from python.utils import parse_args
from python.preprocess import preprocess
from python.run_models import run

if __name__ == "__main__":
    print(sys.argv)
    args = parse_args(sys.argv[1:])
    print(args)

    if args.prep:
        print("Preprocess")
        preprocess(args)
    elif args.run:
        print("Run")
        run(args)
    elif args.gather_results:
        print("Gather results")
    else:
        print("None of the main options selected, print usage")

