# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest
from PIL import Image, ImageDraw

# Import from package
from christmais.drawsys import Artist


def test_artist_draw(embedding):
    """Test if Artist.draw() method returns PIL.Image"""
    artist = Artist(embedding)
    img = artist.draw()
    assert isinstance(img, Image.Image)


def test_artist_draw_circles(embedding, imdraw_canvas):
    """Test private method _draw_circles()"""
    artist = Artist(embedding)
    im, draw = imdraw_canvas
    layers = {
        l: tuple(np.random.randint(0, 256, size=4))
        for l in ["layer1", "layer2", "layer3"]
    }
    im_, draw_ = artist._draw_circles(im, draw, layers)
    assert isinstance(im_, Image.Image)
    assert isinstance(draw_, ImageDraw.ImageDraw)


def test_artist_draw_lines(embedding, imdraw_canvas):
    """Test private method _draw_lines()"""
    artist = Artist(embedding)
    im, draw = imdraw_canvas
    color = tuple(np.random.randint(0, 256, size=4))
    im_, draw_ = artist._draw_lines(im, draw, color, 1)
    assert isinstance(im_, Image.Image)
    assert isinstance(draw_, ImageDraw.ImageDraw)


def test_artist_generate_coords(embedding):
    """Test private method _generate_coords()"""
    artist = Artist(embedding)
    cands = artist._generate_coords(embedding, 10)
    assert cands.shape == (10, 8)


@pytest.mark.parametrize("x", [np.random.uniform(size=8), 0.50, [0.25, 0.5]])
def test_interpolate(embedding, x):
    """Test private method _interpolate()"""
    artist = Artist(embedding)
    artist._interpolate(x)


def test_generate_colors(embedding):
    """Test private method _generate_colors"""
    artist = Artist(embedding)
    colors = artist._generate_colors(embedding)
    assert isinstance(colors, dict)
