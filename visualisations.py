#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First simple sklearn classifier"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import os
import learn1
from matplotlib import pyplot as plt
import Levenshtein  # via https://pypi.python.org/pypi/python-Levenshtein/
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple visualisations of the test/train data')
    parser.add_argument('table', help='Name of in and out of class data to read (e.g. scikit_testtrain_apple)')
    args = parser.parse_args()

    data_dir = "data"
    in_class_name = os.path.join(data_dir, args.table + '_in_class.csv')
    out_class_name = os.path.join(data_dir, args.table + '_out_class.csv')

    in_class_lines = learn1.reader(in_class_name)
    out_class_lines = learn1.reader(out_class_name)

    if True:
        # investigate most frequently repeated tweets in each class
        c_in = Counter(in_class_lines)
        c_out = Counter(out_class_lines)

    # some hard-coded display routines for playing with the data...
    if False:
        plt.figure()
        plt.ion()
        if False:  # histogram of tweet lengths
            lengths_in_class = [len(s) for s in in_class_lines]
            lengths_out_class = [len(s) for s in out_class_lines]
            plt.title("Histogram of tweet lengths for classes in " + args.table)
            plt.xlabel("Bins of tweet lengths")
            plt.ylabel("Counts")
            tweet_lengths = (0, 140)
            filename_pattern = "histogram_tweet_lengths_{}.png"
        # note - tried counting spaces with s.count(" ") but this seems to mirror
        # tweet-length
        if True:  # counting number of capital letters
            lengths_in_class = [Levenshtein.hamming(s, s.lower()) for s in in_class_lines]
            lengths_out_class = [Levenshtein.hamming(s, s.lower()) for s in out_class_lines]
            plt.title("Histogram of number of capitals for classes in " + args.table)
            tweet_lengths = (0, 40)
            filename_pattern = "nbr_capitals_{}.png"
        plt.hist(lengths_in_class, range=tweet_lengths, color="blue", label="in-class", histtype="step")
        plt.hist(lengths_out_class, range=tweet_lengths, color="green", label="out-class", histtype="step")
        UPPER_LEFT = 2
        plt.legend(loc=UPPER_LEFT)
        plt.savefig(filename_pattern.format(args.table))
