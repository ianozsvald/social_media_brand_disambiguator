#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Annotate tweets using OpenCalais"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
import argparse
import config
from ner_apis.opencalais import opencalais_ner
import sql_convenience

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tweet annotator using external NER engine')
    parser.add_argument('keyword', help='Keyword we wish to disambiguate (determines table name and used to filter tweets)')
    parser.add_argument('nerengine', help='NER engine type (only "opencalais" at present)')
    parser.add_argument('--drop', default=False, action="store_true", help='Drops the keyword destination table so we do all annotations again')

    args = parser.parse_args()
    print(args)

    if args.nerengine == "opencalais":
        ner = opencalais_ner.OpenCalaisNER
    else:
        1 / 0

    annotations_table = "annotations_{}".format(args.keyword)
    destination_table = "{}_{}".format(args.nerengine, args.keyword)
    cursor = config.db_conn.cursor()

    if args.drop:
        sql = "DROP TABLE IF EXISTS {}".format(destination_table)
        print("Dropping table: {}".format(sql))
        cursor.execute(sql)
    annotations_table, destination_table = sql_convenience.create_all_tables(args.keyword)

    engine = ner(annotations_table, destination_table)
    engine.annotate_all_messages()
