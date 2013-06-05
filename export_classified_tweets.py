#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Annotate tweets by hand to create a gold standard"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import sys
import sql_convenience
import unicodecsv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tweet annotator')
    parser.add_argument('keyword', help='Keyword we wish to disambiguate (determines table name and used to filter tweets)')
    parser.add_argument('--csv', default=None, help='CSV filename to write to (e.g. output.csv), defaults to stdout')
    args = parser.parse_args()

    if args.csv is None:
        writer_stream = sys.stdout
    else:
        writer_stream = open(args.csv, "w")

    writer = unicodecsv.writer(writer_stream, encoding='utf-8')

    classifications_and_tweets = sql_convenience.extract_classifications_and_tweets(args.keyword)
    for cls, tweet in classifications_and_tweets:
        writer.writerow((cls, tweet))

    if not writer_stream.isatty():
        # close the file (but not stdout if that's what we're using!)
        writer_stream.close()
