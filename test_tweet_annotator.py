#!/usr/bin/env python
"""Tests for start_here"""
# -*- coding: utf-8 -*-
# http://www.python.org/dev/peps/pep-0263/
import unittest
import tweet_annotator
import config
import sql_convenience

# simple fixtures
user1 = {'id': 123, 'name': 'ianozsvald'}
tweet1 = {'text': u'example tweet', 'id': 1, 'created_at': "Tue Mar 09 14:01:21 +0000 2010", 'user': user1}


class Test(unittest.TestCase):
    def setUp(self):
        self.annotations_table = "annotations_apple"
        sql_convenience.create_tables(config.db_conn, self.annotations_table, "table_not_needed_here", force_drop_table=True)

    def test_add_1_annotated_row(self):
        cls = 0
        sql_convenience.insert_tweet(tweet1, cls, config.db_conn, self.annotations_table)

        # now check we have the expected 1 row
        cursor = config.db_conn.cursor()
        sql = "SELECT * FROM {}".format(self.annotations_table)
        cursor.execute(sql)
        all_rows = cursor.fetchall()
        self.assertEqual(len(all_rows), 1, "We expect just 1 row")

        count = tweet_annotator.count_nbr_annotated_rows(config.db_conn, self.annotations_table)
        #from nose.tools import set_trace; set_trace()

        self.assertEqual(count, 1)

if __name__ == "__main__":
    unittest.main()
