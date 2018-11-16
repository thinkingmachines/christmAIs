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
        population=30,
        dims=(224, 224),
        seed=42,
        predictor_kwargs={},
        embedding_kwargs={},
    ):
        """Initialize the class

        Parameters
        ----------
        X : str
            Input sentence to transform into
        population : int (default is 30)
            Number of artists created
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
        self.population = population
        self.dims = dims
        self.seed = seed
        self.predictor_kwargs = predictor_kwargs
        self.embedding_kwargs = embedding_kwargs
        # Create secondary attributes
        self.reset()

    def reset(self):
        """Generate a new set of artists, predictor, and model"""
        # Build a FastText model
        self.model = get_fasttext_pretrained(
            load=True, **self.embedding_kwargs
        )
        self.logger.info('Built FastText model')
        # Transform the string using the FastText model
        self.emb = self.model.transform(self.X)
        self.logger.info('Embedding created for `{}`'.format(self.X))
        # Generate a list of artists
        self.artists = [
            Artist(self.emb, self.dims) for i in range(self.population)
        ]
        self.logger.info('Created {} artists'.format(len(self.artists)))
        # Create a predictor
        self.predictor = Predictor(seed=self.seed, **self.predictor_kwargs)
        self.logger.info(
            'Initialized Predictor with {}'.format(
                self.predictor.models.keys()
            )
        )

    def train(self, target, steps=100, population=30):
        """Train and generate images

        Parameters
        ----------
        target : str
            The target ImageNet class label.
        steps : int (default is 100)
            The number of steps to run the optimization algorithm
        population : int (default is 30)
            Number of artists created
        """
        # imgs = [artist.draw() for artist in self.artists]
        pass

    def _batch_draw(self, genes=None):
        """Draw images from artists

        If genes are supplied, then it uses draw as reference

        Returns
        -------
        list of PIL.Image
            drawn images from Artists
        """
        if genes:
            imgs = [a.draw_from_gene(g) for a, g in zip(genes, self.artists)]
        else:
            imgs = [a.draw() for a in self.artists]
        return imgs

    def _batch_get_genes(self):
        """Obtain the genes from all generated artists

        Returns
        -------
        np.ndarray
            Usually of shape (population, 10, 26)
        """
        genes = [a.get_gene() for a in self.artists]
        return np.stack(genes, axis=0)

    def _fitness_fcn(self, imgs, target):
        """Compute the fitness of the images given a target class label

        Parameter
        ---------
        imgs : list of PIL.Image
            A list of generated images for each Artist
        target : str
            The target ImageNet class label.

        Returns
        -------
        np.ndarray
            An array where each element is the fitness of each Artist
        """
        fitness = [self.predictor.predict(img, target)[0] for img in imgs]
        return np.asarray(fitness)
