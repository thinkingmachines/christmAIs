# -*- coding: utf-8 -*-

"""Tests for christmais.drawer"""

# Import standard library
import glob
import os

# Import modules
import numpy as np
import pytest

# Import from package
from christmais.drawer import Drawer


def test_read_categories(drawer):
    """Test if read_categories returns a list"""
    class_list = drawer._read_categories()
    assert isinstance(class_list, list)


@pytest.mark.parametrize('outfile', ['index'])
@pytest.mark.parametrize('label', ['angel'])
def test_create_index_html(drawer, outfile, label):
    """Test if _create_index_html() creates an output HTML file"""
    drawer._create_index_html(outfile=outfile, label=label)
    html = glob.glob('./**/{}.html'.format(outfile), recursive=True)
    for i in html:
        assert os.path.exists(i)


@pytest.mark.parametrize('outfile', ['index'])
@pytest.mark.parametrize('label', ['angel'])
def test_draw_png(drawer, outfile, label):
    """Test if _draw_png() creates an output 3x3 PNG file"""
    drawer._create_index_html(outfile=outfile, label=label)
    # The name of the index file should be the same as the
    # created png file
    drawer._draw_png(outfile)
    png = glob.glob('./**/{}.png'.format(outfile), recursive=True)
    for i in png:
        assert os.path.exists(i)


@pytest.mark.parametrize('outfile', ['index'])
@pytest.mark.parametrize('label', ['angel'])
def test_draw(drawer, outfile, label):
    """Test if draw() works"""
    drawer.draw(label=label, outfile=outfile)
    html = glob.glob('./**/{}.html'.format(outfile), recursive=True)
    png = glob.glob('./**/{}.png'.format(outfile), recursive=True)
    for h, p in zip(html, png):
        assert os.path.exists(h)
        assert os.path.exists(p)
