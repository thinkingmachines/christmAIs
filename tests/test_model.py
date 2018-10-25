# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from christmais.model import FastTextWrapper, get_fasttext_pretrained


@pytest.parameterize("load", [True, False])
def test_model_load(load):
    """Test if a FastText model is always loaded given any option"""
    pass
