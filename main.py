import argparse
import sys
import os

from python.utils import parse_args
from python.preprocess import preprocess
from python.run_models import run

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.prep:
        print("Preprocess")
        preprocess(args)
    elif args.run:
        print("Run")
        run(args)

