#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""

import unittest
import config
from ner_apis import ner_api_caller
import sql_convenience

        #from nose.tools import set_trace; set_trace()

# simple fixtures
user1 = {'id': 123, 'name': 'ianozsvald'}
tweet1 = {'text': u'example tweet', 'id': 234, 'created_at': "Tue Mar 09 14:01:21 +0000 2010", 'user': user1}
tweet2 = {'text': u'example tweet2', 'id': 235, 'created_at': "Tue Mar 09 14:01:21 +0000 2010", 'user': user1}
tweet3 = {'text': u'example tweet3', 'id': 236, 'created_at': "Tue Mar 09 14:01:21 +0000 2010", 'user': user1}
tweet1class = 0


class Test(unittest.TestCase):
    def setUp(self):
        self.source_table = "annotations_apple"
        self.destination_table = "api_apple"
        sql_convenience.create_tables(config.db_conn, self.source_table, self.destination_table, force_drop_table=True)
        self.ner_api = ner_api_caller.NERAPICaller(self.source_table, self.destination_table)
        self.cursor = config.db_conn.cursor()

    def test1(self):
        """Check we can fetch an unannotated tweet, annotate it and store the result"""
        # check we get a None as we have no messages
        unannotated_message = self.ner_api.get_unannotated_message()
        self.assertEqual(unannotated_message, None)

        # add a tweet to the source_table
        self.insert_tweet(tweet1)
        self.check_we_have_only_n_record(self.source_table, 1)

        # check we get a valid unannotated message
        unannotated_message = self.ner_api.get_unannotated_message()
        self.assertNotEqual(unannotated_message, None)

        #from nose.tools import set_trace; set_trace()
        msg = unannotated_message[str('tweet_text')]
        api_result = self.ner_api.call_api(msg)
        self.assertTrue(api_result.startswith("NERAPICaller base"))

        self.ner_api.store_raw_response(unannotated_message, api_result)
        sql = "SELECT COUNT(*) FROM {}".format(self.destination_table)
        self.cursor.execute(sql)
        all_rows = self.cursor.fetchall()
        #from nose.tools import set_trace; set_trace()
        count = all_rows[0][0]
        self.assertEqual(count, 1, "Check we have 1 new record (we have {})".format(count))

        # check that we cannot store a duplicate item
        self.ner_api.store_raw_response(unannotated_message, api_result)
        sql = "SELECT COUNT(*) FROM {}".format(self.destination_table)
        self.cursor.execute(sql)
        all_rows = self.cursor.fetchall()
        #from nose.tools import set_trace; set_trace()
        count = all_rows[0][0]
        self.assertEqual(count, 1, "Check we have 1 new record (we have {})".format(count))

    def insert_tweet(self, tweet):
        # add a tweet to the source_table
        sql_convenience.insert_tweet(tweet, tweet1class, config.db_conn, self.source_table)

    def check_we_have_only_n_record(self, table, nbr_expected):
        sql = "SELECT COUNT(*) FROM {}".format(table)
        self.cursor.execute(sql)
        all_rows = self.cursor.fetchall()
        count = all_rows[0][0]
        self.assertEqual(count, nbr_expected, "Check we have 1 new record (we have {})".format(count))

    def test2(self):
        """Check that annotate_all_messages does the same job as we've just performed in test1"""
        # add a tweet to the source_table
        self.insert_tweet(tweet1)
        self.check_we_have_only_n_record(self.source_table, 1)
        ## get all tweets from source_table, via annotation, into
        ## destination_table
        self.ner_api.annotate_all_messages()

        sql = "SELECT COUNT(*) FROM {}".format(self.destination_table)
        self.cursor.execute(sql)
        all_rows = self.cursor.fetchall()
        #from nose.tools import set_trace; set_trace()
        count = all_rows[0][0]
        self.assertEqual(count, 1, "Check we have 1 new record (we have {})".format(count))

        self.insert_tweet(tweet2)
        self.insert_tweet(tweet3)
        self.check_we_have_only_n_record(self.source_table, 3)
        ## get all tweets from source_table, via annotation, into
        ## destination_table
        self.ner_api.annotate_all_messages()

        self.check_we_have_only_n_record(self.destination_table, 3)
