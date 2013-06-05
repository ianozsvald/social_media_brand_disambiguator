#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Annotate tweets by hand to create a gold standard"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import config  # assumes env var DISAMBIGUATOR_CONFIG is configured
import tweet_generators
import sql_convenience


def determine_class(tweet):
    """Determine which class our tweet belongs to"""
    tweet_text = unicode(tweet['text'])
    print(tweet_text)
    inp = raw_input("0 for out-of-class, 1 for in-class (i.e. this is the brand), <return> to ignore:")
    cls = sql_convenience.CLASS_UNKNOWN
    if inp.strip() == "0":
        print("out of class")
        cls = sql_convenience.CLASS_OUT
    if inp.strip() == "1":
        print("in class")
        cls = sql_convenience.CLASS_IN
    print("Put into class", cls)
    return cls


def determine_class_and_insert_tweet(tweet, db_conn, annotations_table):
    cls = determine_class(tweet)
    if cls != sql_convenience.CLASS_UNKNOWN:
        sql_convenience.insert_tweet(tweet, cls, db_conn, annotations_table)


def count_nbr_annotated_rows(db_conn, annotations_table):
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM {}".format(annotations_table))
    return cursor.fetchone()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tweet annotator')
    parser.add_argument('tweet_file', help='JSON tweets file for annotation')
    parser.add_argument('keyword', help='Keyword we wish to disambiguate (determines table name and used to filter tweets)')

    args = parser.parse_args()
    print("These are our args:")
    print(args)
    print(args.tweet_file, args.keyword)

    annotations_table, spotlight_table = sql_convenience.create_all_tables(args.keyword)
    tweets = tweet_generators.get_tweets(open(args.tweet_file))

    for tweet in tweets:
        tweet_text = unicode(tweet['text'])
        if args.keyword in tweet_text.lower():
            determine_class_and_insert_tweet(tweet, config.db_conn, annotations_table)
        else:
            print("Skipping tweet due to missing keyword: {}".format(tweet_text))
