# -*- coding: utf-8 -*-

"""Parse a given text to get the nearest QuickDraw class"""

# Import standard library
import logging

# Import modules
import coloredlogs
import gensim.downloader as api
import numpy as np


class Parser:
    """Parser class to get the most similar word"""

    def __init__(
        self, model='glove-wiki-gigaword-50', categories='categories-lite.txt'
    ):
        """Initialize the class with a pretrained FastText model

        Parameters
        ----------
        model : str (default is `glove-wiki-gigaword-50`)
            Model to use
        categories : str (default is 'categories-lite.txt')
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

    def get_most_similar(self, query):
        """Get most similar word based on model

        Parameters
        ----------
        query : str
            Input string

        Returns
        -------
        (str, float)
            Most similar word and similarity score
        """
        query_parts = query.lower().split()
        cat_list = []
        score_list = []
        for query in query_parts:
            try:
                cat, scores = self._get_similar(query)
            except KeyError:
                pass
            else:
                cat_list.append(cat)
                score_list.append(scores)
        sim_label = cat_list[np.argmax(score_list)]
        sim_score = np.max(score_list)
        return sim_label, sim_score

    def _get_similar(self, query):
        scores = [
            self.model.wv.similarity(query, cls) for cls in self.categories
        ]
        sim_label = self.categories[np.argmax(scores)]
        sim_score = np.max(scores)
        return sim_label, sim_score
