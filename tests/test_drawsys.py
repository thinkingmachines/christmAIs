# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest
from PIL import Image, ImageDraw

# Import from package
from christmais.drawsys import Artist


def test_artist_draw_return_type(embedding):
    """Test if Artist.draw() method returns PIL.Image"""
    artist = Artist(embedding)
    img = artist.draw()
    assert isinstance(img, Image.Image)


def test_artist_draw_circles_return_type(embedding, imdraw_canvas):
    """Test private method _draw_circles() return type"""
    artist = Artist(embedding)
    im, draw = imdraw_canvas
    layers = {
        l: tuple(np.random.randint(0, 256, size=4))
        for l in ['layer1', 'layer2', 'layer3']
    }
    im_, draw_ = artist._draw_circles(im, draw, layers)
    assert isinstance(im_, Image.Image)
    assert isinstance(draw_, ImageDraw.ImageDraw)


def test_artist_draw_lines_return_type(embedding, imdraw_canvas):
    """Test private method _draw_lines() return type"""
    artist = Artist(embedding)
    im, draw = imdraw_canvas
    color = tuple(np.random.randint(0, 256, size=4))
    im_, draw_ = artist._draw_lines(im, draw, color, 1)
    assert isinstance(im_, Image.Image)
    assert isinstance(draw_, ImageDraw.ImageDraw)


def test_artist_generate_coords_return_shape(embedding):
    """Test private method _generate_coords() shape"""
    artist = Artist(embedding)
    cands = artist._generate_coords(embedding, 10)
    assert cands.shape == (10, 8)


@pytest.mark.parametrize('x', [np.random.uniform(size=8), 0.50, [0.25, 0.5]])
def test_interpolate_run_without_fail(embedding, x):
    """Test private method _interpolate() if it runs without fail"""
    artist = Artist(embedding)
    artist._interpolate(x)


def test_generate_colors_return_type(embedding):
    """Test private method _generate_colors if it returns the correct type"""
    artist = Artist(embedding)
    colors = artist._generate_colors(embedding)
    assert isinstance(colors, dict)


def test_gene_shape(embedding):
    """Test gene return shape"""
    artist = Artist(embedding)
    artist.draw(density=10)
    gene = artist.get_gene()
    assert gene.shape == (10, 26)


def test_gene_return_values(embedding):
    """Test if gene return values correspond to the actual layers"""
    artist = Artist(embedding)
    artist.draw(density=10)
    gene = artist.get_gene()
    assert (artist._circle_coords['layer1'] == gene[:, 0:6]).all()
    assert (artist._circle_coords['layer2'] == gene[:, 6:12]).all()
    assert (artist._circle_coords['layer3'] == gene[:, 12:18]).all()
    assert (artist._circle_w['layer1'].ravel() == gene[:, 18]).all()
    assert (artist._circle_w['layer2'].ravel() == gene[:, 19]).all()
    assert (artist._circle_w['layer3'].ravel() == gene[:, 20]).all()
    assert (artist._line_coords == gene[:, 21:25]).all()
    assert (artist._line_w.ravel() == gene[:, 25]).all()


def test_draw_from_gene_return_type(embedding):
    """Test if the draw_from_gene() returns a PIL.Image type"""
    artist = Artist(embedding)
    artist.draw(density=10)
    gene = artist.get_gene()
    img = artist.draw_from_gene(gene)
    assert isinstance(img, Image.Image)


def test_draw_from_gene_return_same_img(embedding):
    """Test if draw_from_gene() returns the same image"""
    artist = Artist(embedding)
    ground_truth_img = artist.draw(density=10)
    gene = artist.get_gene()
    test_img = artist.draw_from_gene(gene)
    assert np.allclose(ground_truth_img, test_img)
