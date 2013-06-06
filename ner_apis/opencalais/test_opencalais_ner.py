#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""

import unittest
import config
from ner_apis.opencalais import opencalais_ner
import sql_convenience

ENTITIES1 = [
    {'__reference': 'http://d.opencalais.com/comphash-1/705cd5cf-93e1-323c-8d4e-1ea3200d37e4',
     '_type': 'Company',
     '_typeReference': 'http://s.opencalais.com/1/type/em/e/Company',
     'instances': [{'detection': '[and bought some hair gel, then I went to the ]Apple[ store and bought a Macbook Air and an iPod, iPod]',
                    'exact': 'Apple',
                    'length': 5,
                    'offset': 61,
                    'prefix': 'and bought some hair gel, then I went to the ',
                    'suffix': ' store and bought a Macbook Air and an iPod, iPod'}],
     'name': 'Apple',
     'nationality': 'N/A',
     'relevance': 0.629,
     'resolutions': [{'id': 'http://d.opencalais.com/er/company/ralg-tr1r/23d07771-c50b-315b-8050-3cdaf47ac0d0',
                      'name': 'APPLE INC.',
                      'score': 1,
                      'shortname': 'Apple',
                      'symbol': 'AAPL.OQ',
                      'ticker': 'AAPL'}]},
    {'__reference': 'http://d.opencalais.com/genericHasher-1/3a0f3359-b89a-3959-a958-a9141e8c1f9d',
     '_type': 'Product',
     '_typeReference': 'http://s.opencalais.com/1/type/em/e/Product',
     'instances': [{'detection': '[ bought a Macbook Air and an iPod, iPod Touch and ]iPhone[ 4s]',
                    'exact': 'iPhone',
                    'length': 6,
                    'offset': 126,
                    'prefix': ' bought a Macbook Air and an iPod, iPod Touch and ',
                    'suffix': ' 4s'}],
     'name': 'iPhone',
     'producttype': 'Electronics',
     'relevance': 0.629}
]

# MESSAGE1 includes a unicode string that needs encoding for OpenCalais
MESSAGE1 = u"I wish that Apple iPhones were more fun\U0001f34e"

USER1 = {'id': 123, 'name': 'ianozsvald'}
TWEET1 = {'text': MESSAGE1, 'id': 234, 'created_at': "Tue Mar 09 14:01:21 +0000 2010", 'user': USER1}
TWEET1CLASS = 1


class Test(unittest.TestCase):
    def setUp(self):
        self.source_table = "annotations_apple"
        self.destination_table = "api_apple"
        sql_convenience.create_tables(config.db_conn, self.source_table, self.destination_table, force_drop_table=True)
        self.api = opencalais_ner.OpenCalaisNER(self.source_table, self.destination_table)
        self.cursor = config.db_conn.cursor()

    def test_get_list_of_companies(self):
        list_of_companies = self.api._get_list_of_companies(ENTITIES1)
        self.assertEqual(["Apple"], list_of_companies)

    def test_call_api(self):
        entities = self.api.call_api(MESSAGE1)
        self.assertTrue(len(entities) == 1, "We expect 1 (not {}) items".format(len(entities)))
        list_of_companies = self.api._get_list_of_companies(entities)
        self.assertEqual(["Apple"], list_of_companies)
        #import pdb; pdb.set_trace()

    def check_we_have_only_n_record(self, table, nbr_expected):
        sql = "SELECT COUNT(*) FROM {}".format(table)
        self.cursor.execute(sql)
        all_rows = self.cursor.fetchall()
        count = all_rows[0][0]
        self.assertEqual(count, nbr_expected, "Check we have 1 new record (we have {})".format(count))

    def test_full_loop(self):
        """Check that annotate_all_messages processes all messages"""
        # add a tweet to the source_table
        sql_convenience.insert_tweet(TWEET1, TWEET1CLASS, config.db_conn, self.source_table)

        self.check_we_have_only_n_record(self.source_table, 1)
        ## get all tweets from source_table, via annotation, into
        ## destination_table
        self.api.annotate_all_messages()

        self.assertTrue(self.api.is_brand_of('apple', 234))
        self.assertFalse(self.api.is_brand_of('orange', 234))

