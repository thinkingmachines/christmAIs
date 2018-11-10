# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from christmais.model import FastTextWrapper, get_fasttext_pretrained


@pytest.mark.parametrize('load', [True, False])
def test_model_load(load):
    """Test if a FastTextWrapper class is always loaded given any option"""
    model = get_fasttext_pretrained(load, iter=1)
    assert isinstance(model, FastTextWrapper)


def test_fasttext_return_shape():
    """Test if a transform() returns a vector of shape (8,)"""
    model = get_fasttext_pretrained(load=True, iter=1)
    query = 'Thinking Machines Data Science'
    seed = model.transform(query)
    assert seed.shape == (8,)
    assert isinstance(seed, np.ndarray)
