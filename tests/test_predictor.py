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


@pytest.mark.parametrize("model_list", [["resnet152"], ["resnet152", "vgg16"]])
def test_predictor_get_models(model_list):
    """Test private method _get_models()"""
    predictor = Predictor()
    models = predictor._get_models(model_list)
    assert isinstance(models, dict)


def test_predictor_unrecognized_model():
    """Test private method _get_models() if there is an unrecognized model passed"""
    predictor = Predictor()
    with pytest.raises(KeyError):
        predictor._get_models(["model120"])
