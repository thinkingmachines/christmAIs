# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from christmais.predictor import Predictor


@pytest.mark.parametrize("labels_file", ["labels.json", "extra.json"])
def test_predictor_get_labels(labels_file):
    """Test private method _get_labels()"""
    predictor = Predictor()
    labels = predictor._get_labels("labels.json")
    assert isinstance(labels, dict)
