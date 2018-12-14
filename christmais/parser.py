# -*- coding: utf-8 -*-

"""Parse a given text to get the nearest QuickDraw class"""

# Import standard library
import glob
import logging

# Import modules
import coloredlogs
import gensim.downloader as api
import numpy as np


class Parser:
    """Parser class to get the most similar word"""

    def __init__(
        self, model='glove-wiki-gigaword-50', categories='categories.txt'
    ):
        """Initialize the class with a pretrained FastText model

        Parameters
        ----------
        model : str (default is `glove-wiki-gigaword-50`)
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
        self.categories = self._read_categories(categories)

    def _read_categories(self, category):
        """Read category files

        Parameters
        ----------
        category : str
            Name of categories file
        """
        with open(
            glob.glob('./**/{}'.format(category), recursive=True)[0], 'r'
        ) as fp:
            data = fp.readlines()
        return [d.rstrip() for d in data]

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
                # Handle edge-cases when there's no good word.
                # Use `dog` as category, and set score to something
                # small
                cat = 'dog'
                scores = np.random.uniform(0, 0.3)
            # Append anyway
            cat_list.append(cat)
            score_list.append(scores)
        sim_label = cat_list[np.argmax(score_list)]
        sim_score = np.max(score_list)
        return self._get_actual_label(sim_label), sim_score

    def _get_actual_label(self, label):
        """Get the actual label

        Parameters
        ----------
        label : str
            The input label

        Returns
        -------
        str
            Actual label from Quick, Draw! categories
        """
        # Actual quickdraw names
        qd_names = {
            "clock": "alarm_clock",
            "board": "diving_board",
            "ship": "cruise_ship",
            "hydrant": "fire_hydrant",
            "tree": "palm_tree",
            "outlet": "power_outlet",
            "painting": "the_mona_lisa",
        }
        if label in qd_names.keys():
            self.logger.debug(
                'Converting label to Quick, Draw! compliant labels'
            )
            actual_label = qd_names[label]
        else:
            actual_label = label

        return actual_label

    def _get_similar(self, query):
        """Get similar word for a 1-word query"""
        scores = [
            self.model.wv.similarity(query, cls) for cls in self.categories
        ]
        sim_label = self.categories[np.argmax(scores)]
        sim_score = np.max(scores)
        return sim_label, sim_score
