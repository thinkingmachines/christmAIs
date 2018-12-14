# -*- coding: utf-8 -*-

# Import modules
import glob
import pytest
import tensorflow as tf

# Import from package
from christmais.parser import Parser
from christmais.drawer import Drawer
from christmais.styler import Styler


def pytest_addoption(parser):
    parser.addoption('--local', action='store_true')


@pytest.fixture
def local(request):
    return request.config.getoption('--local')


@pytest.fixture
def parser():
    """Create a pre-made parser"""
    return Parser()


@pytest.fixture
def drawer():
    """Create a pre-made drawer"""
    webdriver = glob.glob('./**/test-chromedriver', recursive=True)
    return Drawer(webdriver[0])


@pytest.fixture
def styler(local):
    """Create a pre-made styler"""
    if local:
        checkpoint = './tests/data/ckpt/model.ckpt'
    else:
        checkpoint = '/usr/src/app/ckpt/model.ckpt'

    return Styler(checkpoint=checkpoint, output='./output')


@pytest.fixture
def process_img():
    def _process_img(path, styler, square_crop, size):
        img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        img_preprocessed = styler._create_placeholder(
            square_crop, img_ph, size
        )
        img_path = glob.glob('./**/{}'.format(path), recursive=True)
        img_np, img_name = styler._get_data_and_name(img_path[0])
        return img_preprocessed, img_ph, img_np, img_name

    return _process_img
