# -*- coding: utf-8 -*-

"""Text-to-abstract art generation for the holidays!"""

from .model import get_fasttext_pretrained, FastTextWrapper
from .drawsys import Artist
from .predictor import Predictor

__all__ = ["get_fasttext_pretrained", "FastTextWrapper", "Artist", "Predictor"]
