#!/usr/bin/env python
# -*- coding: utf-8 -*-
# http://www.python.org/dev/peps/pep-0263/
"""Base class to call Named Entity Recognition APIs"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import random
import json
import config
import sql_convenience


class NERAPICaller(object):
    def __init__(self, source_table, destination_table):
        self.source_table = source_table
        self.destination_table = destination_table
        self.brand = "apple"  # the brand we're testing

    def annotate_all_messages(self):
        while True:
            msg = self.get_unannotated_message()
            if msg is not None:
                tweet_id = msg[b'tweet_id']
                tweet_text = msg[b'tweet_text']
                config.logging.info('Asking API for results for "%r"' % (repr(tweet_text)))
                response = self.call_api(tweet_text)
                self.store_raw_response(msg, response)
                if self.is_brand_of(self.brand, tweet_id):
                    cls = sql_convenience.CLASS_IN
                else:
                    cls = sql_convenience.CLASS_OUT
                # assign class to this tweet
                sql_convenience.update_class(tweet_id, self.destination_table, cls)
            else:
                break

    def get_unannotated_message(self):
        """Return 1 not-yet-annotated message from the source_table"""
        cursor = config.db_conn.cursor()
        # select a tweet where the tweet isn't already in the destination_table
        # http://stackoverflow.com/questions/367863/sql-find-records-from-one-table-which-dont-exist-in-another
        sql = "SELECT tweet_id, tweet_text FROM {} WHERE tweet_id NOT IN (SELECT tweet_id FROM {})".format(self.source_table, self.destination_table)
        cursor.execute(sql)
        all_rows = cursor.fetchall()
        # return a random item or None if there are no messages left
        # unannotated
        if len(all_rows) > 0:
            return random.choice(all_rows)
        else:
            return None

    def call_api(self, message):
        """Return a simulated call to an API"""
        return "NERAPICaller base class:{}".format(message)

    def is_brand_of(self, brand, tweet_id):
        """By default we assume all tweets have no brand in this base class"""
        return False

    def store_raw_response(self, source_details, response_text):
        """Store raw response from API provider using source_details"""
        cls = sql_convenience.CLASS_NOT_INVESTIGATED
        json_response = json.dumps(response_text)
        sql_convenience.insert_api_response(source_details[b'tweet_id'], source_details[b'tweet_text'], json_response, cls, config.db_conn, self.destination_table)
