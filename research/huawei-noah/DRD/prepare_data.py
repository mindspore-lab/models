import os, sys
import getopt

import argparse
from preprocessing.rand_sample import rand_sample
from preprocessing.svm_rank import svm_rank
from preprocessing.output_json import handle_data
from preprocessing.clean_data import clean_data


def main():
    parser = argparse.ArgumentParser(description="prepare datasets")
    parser.add_argument('-f', '--file_path', dest='file_path', required=True)
    # parser.add_argument('-c', '--coefficient', type=float, default=200)
    parser.add_argument('-c', '--coefficient', type=float, default=200)
    parser.add_argument('-p', '--proportion', type=float, default=0.01)
    args = parser.parse_args()
    file_path = args.file_path
    coefficient = args.coefficient
    proportion = args.proportion

    clean_data(file_path)
    rand_sample(file_path, proportion)
    svm_rank(file_path, coefficient)
    handle_data(file_path)


if __name__ == "__main__":
    main()