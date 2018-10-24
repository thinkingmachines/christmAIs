# -*- coding: utf-8 -*-

"""Embed text input into a numerical vector using the FastText model"""


# Import standard library
import logging

# Import modules
import numpy as np
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from nltk.corpus import brown
from sklearn.preprocessing import minmax_scale

logging.basicConfig(level=logging.INFO)


def get_fasttext_pretrained(load=False, **kwargs):
    """Returns a FastTextWrapper model pre-trained with brown corpus

    Parameters
    ----------
    load : bool (default is False)
        Load a trained FastText.model from disk
    **kwargs : dict
        Keyword arguments similar to FastTextWrapper

    Returns
    -------
    model.FastTextWrapper
        A fitted instance of the FastTextWrapper model

    """
    logger = logging.getLogger(__name__)
    if load:
        fname = get_tmpfile("brown_fasttext.model")
        try:
            model = FastTextWrapper.load(fname)
        except FileNotFoundError:
            msg = "{} not found, will train FastText with brown corpus..."
            logger.warn(msg.format(fname))
            model = FastTextWrapper(sentences=brown.sents(), **kwargs)
            model.save(fname)
    else:
        model = FastTextWrapper(sentences=brown.sents(), **kwargs)

    return model


class FastTextWrapper(FastText):
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
        super(FastTextWrapper, self).__init__(
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
                v = self.wv.word_vec(word)
            except KeyError:
                # If `word` is not in hashed corpus, then
                # just generate a random vector
                v = np.random.uniform(size=self.size)
            vector.append(v)
        vector = np.vstack(vector).mean(axis=0)
        vector = minmax_scale(vector, feature_range=(0.02, 0.98))
        return vector
