# -*- coding: utf-8 -*-

"""Tests for the trainer system"""

import pytest
import numpy as np

from christmais.trainer import Trainer


def test_fitness_function_return_type():
    """Test if the fitness function returns the a numpy.ndarray"""
    t = Trainer('Thinking Machines Data Science')
    imgs = t._batch_draw()
    fitness = t._fitness_fcn(imgs, target='iron')
    assert isinstance(fitness, np.ndarray)


def test_fitness_function_return_shape():
    """Test if the fitness function returns the same number of Artists"""
    expected_shape = (30,)
    t = Trainer('Thinking Machines Data Science', population=expected_shape[0])
    imgs = t._batch_draw()
    fitness = t._fitness_fcn(imgs, target='iron')
    assert fitness.shape == expected_shape


def test_batch_draw_return_shape():
    """Test if batch_draw returns the same number of Artists"""
    nb_artists = 30
    t = Trainer('Thinking Machines Data Science', population=nb_artists)
    imgs = t._batch_draw()
    assert len(imgs) == nb_artists


def test_batch_get_genes_return_shape():
    """Test if batch_get_genes return the expected shape"""
    population = 30
    expected_shape = (population, 10, 26)
    t = Trainer('Thinking Machines Data Science', population)
    t._batch_draw()
    genes = t._batch_get_genes()
    assert genes.shape == expected_shape
