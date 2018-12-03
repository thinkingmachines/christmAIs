# -*- coding: utf-8 -*-

"""Tests for christmais.parser"""

# Import standard library
import glob

# Import modules
import numpy as np
import pytest

# Import from package
from christmais.parser import Parser


@pytest.mark.parametrize('cat', glob.glob('**/categories.txt', recursive=True))
def test_read_categories(cat, parser):
    """Test if _read_categories returns a list"""
    class_list = parser._read_categories(cat)
    assert isinstance(class_list, list)


@pytest.mark.parametrize(
    'query', ['Thinking Machines Data Science', 'Lj Miranda', 'Avocado']
)
def test_get_most_similar(query, parser):
    """Test if it returns (str, float) given a set of query strings"""
    label, score = parser.get_most_similar(query)
    assert isinstance(label, str)
    assert isinstance(score, (float, int, np.float32))


@pytest.mark.parametrize('label', [('clock', 'alarm_clock'), ('yoga', 'yoga')])
def test_get_actual_label(label, parser):
    """Test if it returns an equivalent string"""
    query, expected = label
    label = parser._get_actual_label(query)
    assert label == expected


@pytest.mark.parametrize('query', ['yoga', 'book', 'outlet', 'tiger'])
def test_get_similar(query, parser):
    """Test if it returns (str, float) given a set of query strings"""
    label, score = parser._get_similar(query)
    assert isinstance(label, str)
    assert isinstance(score, (float, int, np.float32))
