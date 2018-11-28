# -*- coding: utf-8 -*-

"""Parse a given text to get the nearest QuickDraw class"""

import logging
import coloredlogs
import numpy as np


import gensim.downloader as api


class Parser:
    """Parser class to get the most similar word"""

    def __init__(
        self,
        model='fasttext-wiki-news-subwords-300',
        categories='categories.txt',
    ):
        """Initialize the class with a pretrained FastText model

        Parameters
        ----------
        model : str (default is fasttext-wiki-news-subwords-300)
            Model to use
        categories : str (default is 'categories.txt')
            Location of category strings for Quick, Draw!
        """
        # Initialize the logger
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(logging.INFO, logger=self.logger)
        # Initialize models and classes
        self.logger.info('Initializing model: {}'.format(model))
        self.model = api.load(model)
        self.logger.info('Initializing categories: {}'.format(categories))
        with open(categories, 'r') as fp:
            x = fp.readlines()
        self.categories = [cat.rstrip() for cat in x]
