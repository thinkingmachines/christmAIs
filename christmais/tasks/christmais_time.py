"""Console script for an end-to-end christmAIs pipeline"""

# Import standard library
import logging
from argparse import ArgumentParser

# Import modules
import coloredlogs

# Import from package
from christmais.drawer import Drawer
from christmais.parser import Parser
from christmais.styler import Styler


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        dest='input_query',
        help='input query string',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-s',
        '--style',
        dest='style',
        help='path to style file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-o',
        '--output',
        dest='output',
        help='name of styled output file (no file extension)',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-c',
        '--categories-path',
        dest='categories_path',
        help='path to categories list',
        default='categories.txt',
        type=str,
    )
    parser.add_argument(
        '-m',
        '--model-path',
        dest='model_path',
        help='path to model checkpoint',
        default='./ckpt/model.ckpt',
        type=str,
    )
    parser.add_argument(
        '-d',
        '--webdriver-path',
        dest='webdriver_path',
        help='path to webdriver',
        default='./webdriver/chromedriver',
    )
    return parser


def main():
    # Build argument parser
    parser = build_parser()
    options = parser.parse_args()
    logger = logging.getLogger('christmais_time')
    coloredlogs.install(logging.INFO, logger=logger)

    # Initialize classes
    p = Parser(categories=options.categories_path)
    d = Drawer(webdriver_path=options.webdriver_path)
    s = Styler(checkpoint=options.model_path, output='./artifacts')

    # Pipeline
    ## Get most similar Quick, Draw! class
    logger.info('Finding the nearest class..')
    label, score = p.get_most_similar(query=options.input_query)
    logger.info('Nearest label is {} (score={})'.format(label, score))
    ## Draw the corresponding class in a 3x3 grid
    logger.info('Drawing Quick, Draw! class...')
    d.draw(label=label, outfile=options.output)
    ## Style the output file
    logger.info('Applying style transfer')
    s.style_transfer(content_path=options.output, style_path=options.style)
    logger.info('Done!')


if __name__ == '__main__':
    main()
