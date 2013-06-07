#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare two tables, generate a set of scores"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import sql_convenience

if __name__ == "__main__":
    # gold_std table, comparison_table
    parser = argparse.ArgumentParser(description='Score results against a gold standard')
    parser.add_argument('gold_standard_table', help='Name of the gold standard table (e.g. annotations_apple)')
    parser.add_argument('comparison_table', help='Name of the table we will score against the gold_standard_table (e.g. scikit_apple)')
    args = parser.parse_args()

    # counters for the 4 types of classification
    tp = 0  # True Positives (predicted in class and are actually in class)
    tn = 0  # True Negatives (predicted out of class and are actually out of class)
    fp = 0  # False Positives (predicted in class but are actually out of class)
    fn = 0  # False Negatives (predicted out of class but are actually in class)

    # for each tweet in comparison table, get tweet_id and cls
    classifications_and_tweets = sql_convenience.extract_classifications_and_tweets(args.gold_standard_table)
    for gold_class, tweet_id, tweet in classifications_and_tweets:
        cls, _, _ = sql_convenience.extract_classification_and_tweet(args.comparison_table, tweet_id)
        if gold_class == sql_convenience.CLASS_IN:
            if cls == sql_convenience.CLASS_IN:
                tp += 1
            else:
                assert cls == sql_convenience.CLASS_OUT
                fn += 1
        else:
            assert gold_class == sql_convenience.CLASS_OUT
            if cls == sql_convenience.CLASS_OUT:
                tn += 1
            else:
                assert cls == sql_convenience.CLASS_IN
                fp += 1

    print("True pos {}, False pos {}, True neg {}, False neg {}".format(tp, fp, tn, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision {}, Recall {}".format(precision, recall))
