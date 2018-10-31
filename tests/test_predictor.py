# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from christmais.predictor import Predictor
from christmais.drawsys import Artist


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


@pytest.mark.parametrize("model_list", [["resnet152"], ["resnet152", "vgg16"]])
def test_predict(model_list):
    """Test if predict returns expected results"""
    artist = Artist(np.random.uniform(size=8))
    img = artist.draw()
    p = Predictor(model_list)
    score, results = p.predict(img)
    assert isinstance(score, float)
    assert isinstance(results, dict)
