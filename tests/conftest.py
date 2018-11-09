# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest
from PIL import Image, ImageDraw

# Set random seed
np.random.seed(42)


@pytest.fixture()
def embedding():
    """Random 8-bit embedding"""
    return np.random.uniform(size=8)


@pytest.fixture()
def imdraw_canvas():
    """A basic canvas that returns (Image.Image, ImageDraw)"""
    im = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(im, "RGBA")
    return (im, draw)
