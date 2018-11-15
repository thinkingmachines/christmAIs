# -*- coding: utf-8 -*-

"""Trainer system for integration"""

# Import standard library
import logging

# Import modules
import numpy as np

from .drawsys import Artist
from .embedder import get_fasttext_pretrained
from .predictor import Predictor

logging.basicConfig(level=logging.INFO)


class Trainer:
    """Trainer class for integration"""

    def __init__(
        self,
        X,
        dims=(224, 224),
        seed=42,
        predictor_kwargs=None,
        embedding_kwargs=None,
    ):
        """Initialize the class

        Parameters
        ----------
        X : str
            Input sentence to transform into
        dims : tuple of size 2
            Dimensions of the resulting image
        seed : int (default is 42)
            Sets the random seed
        predictor_kwargs : dict
            Arguments for Predictor() class
        embedding_kwargs : dict
            Arguments for FastTextWrapper() class
        """
        self.logger = logging.getLogger(__name__)
        self.X = X
        self.dims = dims
        self.seed = seed
        self.predictor_kwargs = predictor_kwargs
        self.embedding_kwargs = embedding_kwargs
        # Create secondary attributes
        self.artists = None
        self.model = get_fasttext_pretrained(**embedding_kwargs)
        self.emb = self.model.transform(X)

    def train(self, steps=100, population=30):
        """Train and generate images

        Parameters
        ----------
        steps : int (default is 100)
            The number of steps to run the optimization algorithm
        population : int (default is 30)
            Number of artists created
        """
        self.artists = [Artist(self.emb, self.dims) for i in range(population)]
        self.predictor = Predictor(seed=42, **self.predictor_kwargs)
        # imgs = [artist.draw() for artist in self.artists]
