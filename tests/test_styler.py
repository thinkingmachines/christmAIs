# -*- coding: utf-8 -*-

"""Tests for christmais.styler"""

# Import standard library
import glob

# Import modules
import numpy as np
import pytest
import tensorflow as tf
from christmais.styler import Styler


@pytest.mark.parametrize('square_crop', [True, False])
def test_create_placeholder_return_type(styler, square_crop):
    """Test if _create_placeholder() returns expected type"""
    img_ph = tf.placeholder(tf.float32, [None, None, 3])
    img_ph_resized = styler._create_placeholder(
        square_crop=square_crop, img_ph=img_ph, size=256
    )
    assert isinstance(img_ph_resized, tf.Tensor)


@pytest.mark.parametrize('content_list', ['**/test_content.jpg'])
@pytest.mark.parametrize('style_list', ['**/test_style.jpg'])
def test_get_img_lists_return_type(styler, content_list, style_list):
    """Test if _get_img_lists() returns two lists"""
    c_l, s_l = styler._get_img_lists(content_list, style_list, 1024)
    assert isinstance(c_l, list)
    assert isinstance(s_l, list)


@pytest.mark.parametrize('content_list', ['./**/test_content.jpg'])
@pytest.mark.parametrize('style_list', ['./**/test_style.jpg'])
def test_get_img_lists_return_objects(styler, content_list, style_list):
    """Test if _get_img_lists() actually finds something"""
    c_l, s_l = styler._get_img_lists(content_list, style_list, 1024)
    assert len(c_l) > 0
    assert len(s_l) > 0


@pytest.mark.parametrize('img_path', ['test_content.jpg', 'test_style.jpg'])
def test_get_data_and_name(styler, img_path):
    """Test if _get_data_and_name() returns expected types"""
    img = glob.glob('./**/{}'.format(img_path), recursive=True)
    img_vector, img_name = styler._get_data_and_name(img[0])
    assert isinstance(img_vector, np.ndarray)
    assert isinstance(img_name, str)


@pytest.mark.parametrize('path', ['test_content.jpg', 'test_style.jpg'])
@pytest.mark.parametrize('square_crop', [True, False])
@pytest.mark.parametrize('size', [256, 128, 72])
def test_save_preprocessed_img(styler, process_img, path, square_crop, size):
    """Test if _save_preprocessed_img saves the results"""
    with tf.Graph().as_default(), tf.Session() as sess:
        img_preprocessed, img_ph, img_np, img_name = process_img(
            path=path, styler=styler, square_crop=square_crop, size=size
        )
        styler._save_preprocessed_img(
            sess=sess,
            img_preprocessed=img_preprocessed,
            img_ph=img_ph,
            img_np=img_np,
            img_name=img_name,
        )


@pytest.mark.parametrize('content_path', ['test_content.jpg'])
@pytest.mark.parametrize('style_path', ['test_style.jpg'])
@pytest.mark.parametrize('content_size', [256])
@pytest.mark.parametrize('style_size', [256])
@pytest.mark.parametrize('content_square_crop', [True, False])
@pytest.mark.parametrize('style_square_crop', [True, False])
@pytest.mark.parametrize('interp_weights', [[1.0]])
def test_style_transfer(
    styler,
    content_path,
    style_path,
    content_size,
    style_size,
    content_square_crop,
    style_square_crop,
    interp_weights,
):
    """Test if style_transfer returns no errors inside tf.Session"""
    styler.style_transfer(
        content_path=content_path,
        style_path=style_path,
        content_size=content_size,
        style_size=style_size,
        content_square_crop=content_square_crop,
        style_square_crop=style_square_crop,
        interp_weights=interp_weights,
    )
