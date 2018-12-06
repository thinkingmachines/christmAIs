# -*- coding: utf-8 -*-

"""Utility script for interacting with the Parser class

See Also
--------
:class:`christmais.parser.Parser`: the Parser class for string to nearest QuickDraw class
"""

# Import standard library
import argparse
import logging

# Import modules
from christmais.parser import Parser
# Import from package
import coloredlogs


def build_parser():
    """Create arguments when running the application in console.

    Returns
    -------
    argparse.ArgumentParser
        contains arguments used during program execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--categories',
        dest='categories',
        help='location of category file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        help='model to use',
        type=str,
        default='glove-wiki-gigaword-50'
    )
    parser.add_argument(
        '-q',
        '--query',
        dest='query',
        help='string to query (if there are spaces, enclose in quotation marks)',
        type=str,
        required=True
    )
    return parser


def main():
    """Main routine"""
    # Create logger
    logger = logging.getLogger('christmais.parse_string')
    coloredlogs.install(level=logging.INFO, logger=logger)
    # Get params for parser
    options = build_parser()
    # Initialize parser and do things
    p = Parser(model=options.model, categories=options.categories)
    quick_draw_class = p.get_most_similar(options.query)
    logger.info('The nearest quick draw class: {}'.format(quick_draw_class))
    return quick_draw_class


if __name__ == '__main__':
    main()
