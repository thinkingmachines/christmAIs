# -*- coding: utf-8 -*-

"""Console script to generate abstract art"""


# Import standard library
import json
import logging
from argparse import ArgumentParser

# Import modules
import coloredlogs
import numpy as np
import torch

# Import from package
from christmais import Trainer

logger = logging.getLogger(__name__)
coloredlogs.install(logging.INFO, logger=logger)


def build_parser():
    """Create arguments when running the application in console.

    Returns
    -------
    parser : ArgumentParser
        Contains arguments used during program execution
    """
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input-str',
        dest='input_str',
        help='input string',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-t',
        '--target',
        dest='target',
        help='Target ImageNet class',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='output_dir',
        help='Output directory of the best gene or image',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-S',
        '--steps',
        dest='steps',
        help='No. of steps (default is 100)',
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        '-c',
        '--colorscheme',
        dest='colorscheme',
        help='JSON filepath for set colorscheme',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '-O',
        '--train-output-dir',
        dest='train_output_dir',
        help='output directory to store img per gen',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '-P',
        '--pool-size',
        dest='pool_size',
        help='Pool size for mutation (default is 30)',
        type=int,
        default=30,
        required=False,
    )
    parser.add_argument(
        '-p',
        '--population',
        dest='population',
        help='Gene population (default is 30)',
        type=int,
        default=30,
        required=False,
    )
    parser.add_argument(
        '-d',
        '--dimensions',
        dest='dimensions',
        help='Image dimensions to create (default is [400, 400])',
        type=int,
        default=[400, 400],
        required=False,
    )
    parser.add_argument(
        '-m',
        '--mutpb',
        dest='mutpb',
        help='Mutation probability (default is 0.3)',
        type=float,
        default=0.3,
        required=False,
    )
    parser.add_argument(
        '-I',
        '--indpb',
        dest='indpb',
        help='Independent probability for shuffles (default is 0.5)',
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        '-s',
        '--tournsize',
        dest='tournsize',
        help='Tournament selection size (default is 4)',
        type=int,
        default=4,
        required=False,
    )
    return parser


def main():
    logger.debug('Running christmais_time script')
    try:
        device = torch.cuda.get_device_name(0)
        logger.info('Using GPU: {}'.format(device))
    except AssertionError:
        logger.warning('NVIDIA/CUDA-enabled GPU not found. Using CPU')

    parser = build_parser()
    options = parser.parse_args()
    # Create a Trainer
    t = Trainer(
        X=options.input_str,
        population=options.population,
        pool_size=options.pool_size,
        dims=options.dimensions,
    )
    # Set colorscheme and dimensions
    if options.colorscheme is not None:
        with open(options.colorscheme, 'r') as fp:
            colorscheme = json.load(fp)
        logger.debug('Converting lists into tuples')
        for k, v in colorscheme.items():
            colorscheme[k] = tuple(v)
        t.set_colors(colorscheme=colorscheme)
    # t.set_dims(options.dimensions)
    # Perform optimization
    best = t.train(
        target=options.target,
        mutpb=options.mutpb,
        indpb=options.indpb,
        tournsize=options.tournsize,
        steps=options.steps,
        outdir=options.train_output_dir,
    )
    # Save best image
    best_gene = best.gene
    with open('best_gene.txt', 'w') as fp:
        np.savetxt(fp, best_gene)
        logger.info('Saved best_gene to `best_gene.txt`')
    best_img = best.artist.draw_from_gene(best_gene)
    try:
        best_img.save(options.output_dir)
    except ValueError:
        msg = (
            'Please add a file type extension [.png, .jpg] to output file: {}'
        )
        logger.exception(msg.format(options.output_dir))
        raise
    else:
        msg = 'Best gene saved at {}'
        logger.info(msg.format(options.output_dir))


if __name__ == "__main__":
    main()
