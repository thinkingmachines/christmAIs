# -*- coding: utf-8 -*-

"""Embed text input into a numerical vector using the FastText model"""


# Import modules
import numpy as np
from gensim.models import FastText
from nltk.corpus import brown
from sklearn.preprocessing import minmax_scale


def get_fasttext_pretrained(**kwargs):
    """Returns a FastTextWrapper model pre-trained with brown corpus

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments similar to FastTextWrapper

    Returns
    -------
    model.FastTextWrapper
        A fitted instance of the FastTextWrapper model

    """
    return FastTextWrapper(sentences=brown.sents(), **kwargs)


class FastTextWrapper(object):
    """A wrapper for the FastText model from the GenSim library"""

    def __init__(
        self, sentences, size=8, window=5, min_count=5, seed=42, **kwargs
    ):
        """Initialize the model

        Due to GenSim's API, the model is automatically fitted with the given
        corpus once initialized.

        Parameters
        ----------
        sentences : itreable of list of str
            Can simply be a list of lists of tokens
        size : int (default is 8)
            Dimensionality of the word vector
        window : int (default is 5)
            Maximum distance between the curent and predicted word within a
            sentence
        min_count : int (default is 5)
            Ignores all words with total frequency lower than this value
        seed : int (default is 42)
            Sets the random seed
        **kwargs : dict
            Keyword arguments
        """
        self.size = size
        self.seed = seed
        self.model = FastText(
            sentences=sentences,
            size=size,
            window=window,
            min_count=min_count,
            seed=seed,
            **kwargs
        )
        # Set random seed
        np.random.seed(self.seed)

    def transform(self, X):
        """Transform a sentence into its vector representation

        Parameters
        ----------
        X : str
            Input sentence to transform into

        Returns
        -------
        np.ndarray
            Numerical array of size (`size`, )
        """
        sentence = X.split()
        vector = []
        for word in sentence:
            try:
                # Obtain vector from FastText model
                v = self.model.wv.word_vec(word)
            except KeyError:
                # If `word` is not in hashed corpus, then
                # just generate a random vector
                v = np.random.uniform(size=self.size)
            vector.append(v)
        vector = np.vstack(vector).mean(axis=0)
        vector = minmax_scale(vector, feature_range=(0.02, 0.98))
        return vector
