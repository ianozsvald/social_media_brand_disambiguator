#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy a set of rows to a new table in sqlite"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import random
import config
import sql_convenience


def copy_data_to_subsets(cls, source_table, nbr_testtrain, testtrain_table, nbr_validation, validation_table, drop, config):
    cursor = config.db_conn.cursor()
    sql = "SELECT * FROM {} WHERE class=={}".format(source_table, cls)
    cursor.execute(sql)
    rows = cursor.fetchall()
    random.shuffle(rows)
    rows_validation = rows[:nbr_validation]
    rows_traintest = rows[nbr_validation:nbr_validation + nbr_testtrain]

    #print(rows_traintest[0][b'tweet_id'])
    #print(rows_validation[0][b'tweet_id'])
    # move this n using sql to a new table

    sql_convenience.create_results_table(config.db_conn, testtrain_table, force_drop_table=drop)
    sql_convenience.create_results_table(config.db_conn, validation_table, force_drop_table=drop)

    for row in rows_traintest:
        sql_convenience.insert_api_response(row[b'tweet_id'], row[b'tweet_text'], "", cls, config.db_conn, testtrain_table)
    for row in rows_validation:
        sql_convenience.insert_api_response(row[b'tweet_id'], row[b'tweet_text'], "", cls, config.db_conn, validation_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tweet annotator')
    parser.add_argument('source_table')
    parser.add_argument('nbr_testtrain', type=int, help="Number of rows to be copied for testing/training")
    parser.add_argument('testtrain_table')
    parser.add_argument('nbr_validation', type=int, help="Number of rows to be copied for validation")
    parser.add_argument('validation_table')
    parser.add_argument('--drop', default=False, action="store_true", help="If added then testtrain_table and validation_table and dropped before the copies")
    args = parser.parse_args()

    # load all ids for a class (table_name, class)
    cls = 0
    copy_data_to_subsets(cls, args.source_table, args.nbr_testtrain, args.testtrain_table, args.nbr_validation, args.validation_table, args.drop, config)
    cls = 1
    copy_data_to_subsets(cls, args.source_table, args.nbr_testtrain, args.testtrain_table, args.nbr_validation, args.validation_table, False, config)
