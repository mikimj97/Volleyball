import argparse
import sys
import os

from python.utils import parse_args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.prep:
        print("Preprocess")
    elif args.run:
        print("Run")

