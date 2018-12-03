# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from package
from christmais.parser import Parser


@pytest.fixture
def parser():
    """Create a pre-made parser"""
    return Parser()
