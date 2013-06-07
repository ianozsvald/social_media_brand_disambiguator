#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export in-class and out-class tweets to separate files as data for ML system"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import os
import unicodecsv
import sql_convenience


def writer(class_name, table, cls_to_accept):
    class_writer = unicodecsv.writer(open(class_name, 'w'), encoding='utf-8')
    class_writer.writerow(("tweet_id", "tweet_text"))
    for cls, tweet_id, tweet_text in sql_convenience.extract_classifications_and_tweets(args.table):
        if cls == cls_to_accept:
            # remove carriage returns
            tweet_text = tweet_text.replace("\n", " ")
            class_writer.writerow((tweet_id, tweet_text))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score results against a gold standard')
    parser.add_argument('table', help='Name of table to export (e.g. annotations_apple)')
    args = parser.parse_args()

    data_dir = "data"
    in_class_name = os.path.join(data_dir, args.table + '_in_class.csv')
    out_class_name = os.path.join(data_dir, args.table + '_out_class.csv')

    print("Writing in-class examples to: {}".format(in_class_name))
    writer(in_class_name, args.table, sql_convenience.CLASS_IN)
    print("Writing out-of-class examples to: {}".format(out_class_name))
    writer(out_class_name, args.table, sql_convenience.CLASS_OUT)
