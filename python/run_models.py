from sklearn.ensemble import RandomForestClassifier
from python.utils import read_file, write_file

def run(args):
    print("Running...")

    df = read_file(args.input_dir)

    clf = RandomForestClassifier()

    write_file(clf, args.output_dir)
