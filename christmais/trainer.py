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
        artist_kwargs=None,
    ):
        """Initialize the class

        Parameters
        ----------
        X : str
            Input sentence to transform into
        """
        self.X = X
