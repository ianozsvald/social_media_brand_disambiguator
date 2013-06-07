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
import cld  # https://pypi.python.org/pypi/chromium_compact_language_detector


def determine_class(tweet, keyword):
    """Determine which class our tweet belongs to"""
    tweet_text = unicode(tweet['text'])
    GREEN_COLOUR = '\033[92m'
    END_COLOUR = '\033[0m'
    coloured_keyword = GREEN_COLOUR + keyword + END_COLOUR  # colour the keyword green
    #coloured_tweet_text = tweet_text.replace(keyword, coloured_keyword)

    import re
    sub = re.compile(re.escape(keyword), re.IGNORECASE)
    coloured_tweet_text = sub.sub(coloured_keyword, tweet_text)
    print("--------")
    print(tweet_text)
    print(coloured_tweet_text)
    inp = raw_input("0 for out-of-class, {}1 for in-class (i.e. this is the brand){},\n<return> to ignore (e.g. for non-English or irrelevant tweets):".format(GREEN_COLOUR, END_COLOUR))
    cls = sql_convenience.CLASS_UNKNOWN
    if inp.strip() == "0":
        print("out of class")
        cls = sql_convenience.CLASS_OUT
    if inp.strip() == "1":
        print("in class")
        cls = sql_convenience.CLASS_IN
    #print("Put into class", cls)
    return cls


def determine_class_and_insert_tweet(tweet, db_conn, annotations_table, keyword):
    cls = determine_class(tweet, keyword)
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
    parser.add_argument('--skipto', default=None, type=int, help="Skip forwards to this tweet id, continue from the next tweet")
    args = parser.parse_args()
    print("These are our args:")
    print(args)
    print(args.tweet_file, args.keyword)

    annotations_table, spotlight_table = sql_convenience.create_all_tables(args.keyword)
    tweets = tweet_generators.get_tweets(open(args.tweet_file))

    # we can skip through Tweets we've already seen in the same file by
    # specifying a tweet id to jump to
    if args.skipto is not None:
        for tweet in tweets:
            if tweet['id'] == args.skipto:
                break  # continue after this tweet

    for tweet in tweets:
        tweet_text = unicode(tweet['text'])
        annotate = True
        # determine if this is an English tweet or not
        tweet_text_bytesutf8 = tweet_text.encode('utf-8')
        language_name, language_code, is_reliable, text_bytes_found, details = cld.detect(tweet_text_bytesutf8)
        # example: ('SPANISH', 'es', True, 69, [('SPANISH', 'es', 100, 93.45794392523365)])
        print("---")
        print(language_name, language_code, is_reliable)
        if language_code not in set(["en", "un"]):
            annotate = False

        tweet_id = tweet['id']
        if sql_convenience.check_if_tweet_exists(tweet_id, annotations_table) == 0:
            # check our keyword is present as Twitter can provide tweets 'relevant
            # to your keyword' which don't actually contain the keyword (but it
            # might be linked in a t.co title or body text)
            nbr_keywords = tweet_text.lower().count(args.keyword)
            nbr_keywords_hash = tweet_text.lower().count("#" + args.keyword)
            print(nbr_keywords, nbr_keywords_hash)
            if nbr_keywords == nbr_keywords_hash:
                annotate = False
            if annotate:
                determine_class_and_insert_tweet(tweet, config.db_conn, annotations_table, args.keyword)
