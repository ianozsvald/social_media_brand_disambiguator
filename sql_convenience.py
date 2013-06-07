#!/usr/bin/env python
# -*- coding: utf-8 -*-
# http://www.python.org/dev/peps/pep-0263/
"""Base class to call Named Entity Recognition APIs"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import sqlite3
import datetime
import json
from dateutil import parser as dt_parser
import config

CLASS_OUT = 0  # out-of-class (not what we want to learn)
CLASS_IN = 1  # in-class (what we want to learn)
CLASS_UNKNOWN = 2  # investigated but label not chosen
CLASS_MIXED = 3  # usage has more than 1 class
CLASS_NOT_INVESTIGATED = None  # default before we assign one of 0..3


def create_all_tables(keyword):
    """Entry point to setup tables"""
    annotations_table = "annotations_{}".format(keyword)
    opencalais_table = "opencalais_{}".format(keyword)
    create_tables(config.db_conn, annotations_table, opencalais_table)
    return annotations_table, opencalais_table


def create_tables(db_conn, annotations_table, opencalais_table, force_drop_table=False):
    cursor = db_conn.cursor()
    if force_drop_table:
        # drop table if we don't need it
        for table_name in [annotations_table, opencalais_table]:
            sql = "DROP TABLE IF EXISTS {}".format(table_name)
            cursor.execute(sql)
        db_conn.commit()

    # Odd - large ints are not displayed correctly in SQLite Manager but they
    # are in sqlite (cmd line and via python)
    # Hah! Bug 3 for SQLite Manager (Firefox) states that the problem is with
    # big integers in JavaScript!
    # https://code.google.com/p/sqlite-manager/issues/detail?can=2&start=0&num=100&q=&colspec=ID%20Type%20Status%20Priority%20Milestone%20Owner%20Summary&groupby=&sort=&id=3
    # so there was nothing wrong with my code. I'm leaving this note here as a
    # reminder to myself. An example large tweet_id int would be 306093154619240448
    sql = "CREATE TABLE IF NOT EXISTS {} (tweet_id INTEGER UNIQUE, tweet_text TEXT, tweet_created_at DATE, class INT, user_id INT, user_name TEXT)".format(annotations_table)
    cursor.execute(sql)
    sql = "CREATE TABLE IF NOT EXISTS {} (tweet_id INTEGER UNIQUE, tweet_text TEXT, response_fetched_at DATE, class INT, response TEXT)".format(opencalais_table)
    cursor.execute(sql)
    db_conn.commit()


def extract_classification_and_tweet(table, tweet_id):
    """Return the desired tuple (classification, tweet_id, tweet) in table"""
    cursor = config.db_conn.cursor()
    sql = "SELECT * FROM {} WHERE tweet_id=={}".format(table, tweet_id)
    cursor.execute(sql)
    result = cursor.fetchone()
    return (result[b'class'], result[b'tweet_id'], result[b'tweet_text'])


def extract_classifications_and_tweets(table):
    """Yield list of tuples of (classification, tweet_id, tweet) in table"""
    cursor = config.db_conn.cursor()
    sql = "SELECT * FROM {} ORDER BY tweet_id".format(table)
    cursor.execute(sql)
    results = cursor.fetchall()
    for result in results:
        yield(result[b'class'], result[b'tweet_id'], result[b'tweet_text'])


def check_if_tweet_exists(tweet_id, table):
    """Check if the specified tweet_id exists in our table"""
    cursor = config.db_conn.cursor()
    sql = "SELECT count(*) FROM {} WHERE tweet_id=={}".format(table, tweet_id)
    cursor.execute(sql)
    result = cursor.fetchone()
    count = result[b'count(*)']
    return count

def insert_tweet(tweet, cls, db_conn, annotations_table):
    """Insert tweet into database"""
    tweet_id = tweet['id']
    config.logging.info("Inserting tweet_id '{}'".format(tweet_id))
    tweet_text = unicode(tweet['text'])
    user_id = tweet['user']['id']
    user_name = tweet['user']['name'].lower()
    tweet_created_at = dt_parser.parse(tweet['created_at'])
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO {}(tweet_id, tweet_text, tweet_created_at, class, user_id, user_name) values (?, ?, ?, ?, ?, ?)".format(annotations_table),
                   (tweet_id, tweet_text, tweet_created_at, cls, user_id, user_name))
    db_conn.commit()


def insert_api_response(tweet_id, tweet_text, response, cls, db_conn, destination_table):
    """Insert api response into database"""
    try:
        response_fetched_at = datetime.datetime.utcnow()
        cursor = config.db_conn.cursor()
        cursor.execute("INSERT INTO {}(tweet_id, tweet_text, response_fetched_at, class, response) values (?, ?, ?, ?, ?)".format(destination_table),
                       (tweet_id, tweet_text, response_fetched_at, cls, response))
        config.db_conn.commit()
    except sqlite3.IntegrityError:
        pass  # ignore duplicate insert errors (as we're expecting to run with >1 process)


def deserialise_response(tweet_id, table):
    """Get serialised response from table, deserialise the JSON"""
    cursor = config.db_conn.cursor()
    sql = "SELECT * FROM {} WHERE tweet_id=={}".format(table, tweet_id)
    cursor.execute(sql)
    result = cursor.fetchone()
    result_dict = json.loads(result[b'response'])
    return result_dict


def update_class(tweet_id, table, cls):
    """Update the class for tweet_id"""
    cursor = config.db_conn.cursor()
    sql = "UPDATE {} SET class={} WHERE tweet_id=={}".format(table, cls, tweet_id)
    cursor.execute(sql)
    config.db_conn.commit()
