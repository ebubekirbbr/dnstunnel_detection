import argparse
import json
import os
import random
import subprocess
import traceback

import numpy as np
from tqdm import tqdm
import pandas as pd


class AnomalyDataProcessor:
    def __init__(self, filename, batch_size=10, read_line=None, features=None):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.read_line = read_line
        self.features = features
        self.idx = 0
        if read_line:
            self.row_count = read_line
        else:
            self.row_count = self.get_row_count(self.filename)

        self.file = open(self.filename, 'r')

        self.idx = 0

    def get_all_data(self, features):
        data = None
        labels = None

        return data, labels

    def get_row_count(self, filename):
        try:
            result = subprocess.check_output(f"wc -l {filename}", shell=True).decode("utf-8").strip()
            result = result.split(" ")

            row_count = min(self.read_line, int(result[0])) if self.read_line else int(result[0])
            print(f"{self.filename} has {row_count} row.")

        except:
            print("file row could not be fecthed")
            row_count = 0
        return row_count

    def __len__(self):

        if self.read_line:
            return int(self.read_line / self.batch_size) + 1
        else:
            return int(self.row_count / self.batch_size) + 1

    def __iter__(self):
        """
        Initializes the iterator.
        """
        return self

    def __next__(self):
        if self.idx >= self.row_count:
            self.file.close()
            raise StopIteration

        try:
            if self.idx == 0:
                df = pd.read_csv(self.filename, nrows=self.batch_size, header=0)
                self.columns = df.columns.tolist()
            else:
                df = pd.read_csv(
                    self.filename,
                    skiprows=self.idx + 1,  # +1 çünkü header’ı da atlamalıyız
                    nrows=self.batch_size,
                    header=None
                )
                df.columns = self.columns

            if df.empty:
                self.file.close()
                raise StopIteration

            self.idx += len(df)

            label_col = 'label'
            labels = df[label_col].map({
                'safe': 0,
                'dnstunnel': 1
            })

            if self.features:
                X_train = df[self.features]
            else:
                X_train = df

            return X_train, labels

        except Exception as e:
            traceback.print_exc()
            self.file.close()
            raise StopIteration


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function", required=False, help='function name', type=str, default="main")
    parser.add_argument("--file", required=False, help='file', type=str, default=None)
    parser.add_argument("-pr", "--process", required=False, help='process', type=int, default=1)
    parser.add_argument("--dtype", required=False, help='dtype', type=str, default=None)
    parser.add_argument("-ps", "--part_size", required=False, help='part size', type=int, default=100)
    parser.add_argument("-t", "--testing", required=False, help='testing', action="store_true")
    args = parser.parse_args()

    return args


def main():

    args = argument_parsing()


if __name__ == "__main__":
    main()
