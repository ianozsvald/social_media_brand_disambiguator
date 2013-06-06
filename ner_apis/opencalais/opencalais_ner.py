#!/usr/bin/env python
# -*- coding: utf-8 -*-
# http://www.python.org/dev/peps/pep-0263/
"""Call OpenCalais NER"""
from __future__ import division  # 1/2 == 0.5, as in Py3
from __future__ import absolute_import  # avoid hiding global modules with locals
from __future__ import print_function  # force use of print("hello")
from __future__ import unicode_literals  # force unadorned strings "" to be unicode without prepending u""
from ..ner_api_caller import NERAPICaller
from . import calais
import sql_convenience
import config
from . import opencalais_key  # you need to add a 1 line file containing API_KEY=<key>


class OpenCalaisNER(NERAPICaller):
    def __init__(self, source_table, destination_table):
        super(OpenCalaisNER, self).__init__(source_table, destination_table)
        # Create an OpenCalais object.
        self.api = calais.Calais(opencalais_key.API_KEY, submitter="python-calais demo")

    def call_api(self, message):
        """Return dict of results from an NER call to OpenCalais"""
        message_utf8_string = message.encode('utf-8', 'replace')
        entities = []
        try:
            result = self.api.analyze(message_utf8_string)
            try:
                entities = result.entities
            except AttributeError:
                pass
        except ValueError as err:
            config.logging.error("OpenCalais reports %r" % (repr(err)))
        return entities

    def _get_list_of_companies(self, response):
        """Process response, extract companies"""
        companies = []
        for entity in response:
            if entity['_type'] == 'Company':
                companies.append(entity['name'])
        return companies

    def is_brand_of(self, brand_to_check, tweet_id):
        """Does tweet_id's tweet_text reference brand_to_check as the Company name?"""
        # deserialise_response
        response = sql_convenience.deserialise_response(tweet_id, self.destination_table)
        companies = self._get_list_of_companies(response)
        is_in_list = brand_to_check in set([item.lower() for item in companies])
        return is_in_list
