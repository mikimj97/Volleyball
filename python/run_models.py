import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from python.utils import read_file, write_file
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
from python.constants import JUMP_TYPES


def run(args):
    print("Running...")

    f1s = []

    # for i in range():

    df = pd.read_csv(os.path.join(args.input_dir, "Preprocessed_{}_{}.csv".format(
        args.window_size, args.sampling_interval)), index_col=False)
    df.fillna(-9999, inplace=True)

    if args.jumps_only:
        jump_types = JUMP_TYPES["run"]
        df = df[df.iloc[:, -2] != 0]
        df.iloc[:, -2] -= 1
    else:
        jump_types = JUMP_TYPES["prep"]

    # preds = df.iloc[:, -1]

    for i in range(11):

        clf = RandomForestClassifier()
        # clf = AdaBoostClassifier()
        # clf = DecisionTreeClassifier()
        # clf = KNeighborsClassifier(n_neighbors=1000)
        # clf = MLPClassifier(max_iter=1000000)
        # clf = LogisticRegression(max_iter=10000)

        # Do the stuff

    # train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, -2])

        train = df[df.iloc[:, -1] != i]
        test = df[df.iloc[:, -1] == i]

        X_train, y_train = train.drop(train.columns[-2:], axis=1), train.iloc[:, -2]
        y_train = y_train.astype('int')
        X_test, y_test = test.drop(test.columns[-2:], axis=1), test.iloc[:, -2]
        y_test = y_test.astype('int')

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        accuracy = accuracy_score(pred, y_test)
        print("A: {}".format(accuracy))

        f1 = f1_score(y_test, pred, average="macro")
        print("F: {}".format(f1))

        confusion = confusion_matrix(y_test, pred, labels=list(range(len(jump_types))))
        fig = plt.figure()
        plt.matshow(confusion)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        # plt.savefig(os.path.join(args.output_dir,
        #                          "confusion_matrix-{}-{}.png".format(args.window_size, args.sampling_interval)))
        # plt.show()
        plt.close('all')
        # with open(os.path.join(args.output_dir,
        #                        "confusion_matrix-{}-{}.txt".format(args.window_size, args.sampling_interval)), 'w') as f:
        #     f.write(np.array2string(confusion, separator=', '))

        # params = {"Accuracy: ": accuracy, "F1-score: ": f1, "Window size: ": 400, "Overlap: ": 50}

        # date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # Somehow include the classifier as well, and other stuff that could be missing
        # write_file(json.dumps(params), os.path.join(args.output_dir, "RF-{}.txt".format(date)))

        # with open(os.path.join(args.output_dir,
        #                        "volleyball-waist-{}-{}.pkl".format(args.window_size, args.sampling_interval)), 'wb') as f:
        #     pickle.dump(clf, f)

        # imp = clf.feature_importances_
        #
        # imp_dict = {}
        #
        # for key, val in zip(train.columns, imp):
        #     imp_dict[key] = val

        # for i in sorted(imp_dict, key=imp_dict.get, reverse=True):
        #     print(i, imp_dict[i])

        f1s.append(f1)

    avg_f1 = np.average(f1s)
    print("Average: {}".format(avg_f1))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    write_file(str(avg_f1), os.path.join(args.output_dir, "results_{}_{}.txt".format(args.window_size, args.sampling_interval)))
