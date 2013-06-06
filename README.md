 social_media_brand_disambiguator
=================================

Brand disambiguator for tweets to differentiate e.g. Orange vs orange (brand vs foodstuff), using NLTK and scikit-learn

NOTE this is a work in progress, started June 2013, currently it doesn't do very much at all

I'm only using English tweets and only annotating the ones I can clearly differentiate.

Setup
-----

The code runs with Python 2.7 using sqlite.

    $ pip install requirements.txt  # base libs
    $ pip install requirements_matplotlib.txt  # get around MEP11 with second requirements file

Note that generally I'm using Python 3 __future__ imports, the code isn't tested with Python 3 but the porting should be straight-forward. sqlite only wants byte/strings for key indexing (not unicode strings).

Tests
-----

    $ nosetests  # the project defaults to 'testing' if DISAMBIGUATOR_CONFIG isn't set
    $ nosetests -s  # runs without capturing stdout, useful if you're using `import pdb; pdb.set_trace()` for debugging
    $ nosetests --with-coverage --cover-html  # with an HTML coverage report to cover/index.html

Creating a gold standard
------------------------

    $ u'/home/ian/workspace/virtualenvs/tweet_disambiguation_project/prototype1/src'
    $ export DISAMBIGUATOR_CONFIG=production
    $ %run tweet_annotator.py ../../apple10.json apple
    # or
    $ DISAMBIGUATOR_CONFIG=production python tweet_annotator.py ../../apple10.json apple

Annotating the tweets using OpenCalais
--------------------------------------

    $ DISAMBIGUATOR_CONFIG=production python ner_annotator.py apple opencalais --drop  # optionally drop the destination table so we start afresh
    $ DISAMBIGUATOR_CONFIG=production python ner_annotator.py apple opencalais # run in another window to double fetching speed


Exporting results
-----------------

Output an ordered list of classifications and tweets (by tweet_id), allows a diff (e.g. using meld):

    $ DISAMBIGUATOR_CONFIG=production python export_classified_tweets.py annotations_apple > output/annotations_apple.csv
    $ DISAMBIGUATOR_CONFIG=production python export_classified_tweets.py opencalais_apple > output/opencalais_apple.csv

Design flaws
------------

  * sqlite table structure assumes 1 brand per table (e.g. annotations_apple with class set to is-apple-brand or is-apple-somethingelse), this isn't normal form but is probably fine for the prototype

Other notes
-----------
  
  * https://github.com/twitter/twitter-text-rb/blob/master/lib/twitter-text/regex.rb  Notes from twitter on how they handle unicode:
  * http://nerd.eurecom.fr/documentation  possibly worth considering these other APIs?
  * consider replacing activestate xml_to_dict with https://pypi.python.org/pypi/xmldict/0.4.1


3rd party bugs
--------------

SQLite Manager (0.8) in Firefox suffers from a large-int rounding bug due to JavaScript, large ints like Twitter Ids are rounded on display/entry (but they're correctly handled by sqlite3 in Python and sqlite3 at the cmd line): https://code.google.com/p/sqlite-manager/issues/detail?can=2&start=0&num=100&q=&colspec=ID%20Type%20Status%20Priority%20Milestone%20Owner%20Summary&groupby=&sort=&id=3

License
=======

MIT

Copyright (c) 2013 Ian Ozsvald

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentati
on files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom 
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Sof
tware.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WAR
RANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYR
IGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, A
RISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c) 2013 Ian Ozsvald

