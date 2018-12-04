# -*- coding: utf-8 -*-

# Import modules
import glob
import pytest

# Import from package
from christmais.parser import Parser
from christmais.styler import Styler


@pytest.fixture
def parser():
    """Create a pre-made parser"""
    return Parser()

@pytest.fixture
def styler():
    """Create a pre-made styler"""
    return Styler(checkpoint=glob.glob('**/'))
