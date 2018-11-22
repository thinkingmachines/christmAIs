# -*- coding: utf-8 -*-

"""Console script to generate abstract art"""


# Import standard library
from argparse import ArgumentParser

from christmais import Trainer


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
        '-c',
        '--colorscheme',
        dest='colorscheme',
        help='JSON filepath for set colorscheme',
        type=str,
        default=None,
        required=False,
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
        '-O',
        '--train-output-dir',
        dest='train_output_dir',
        help='output directory to store img per gen',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '-S',
        '--steps',
        dest='steps',
        help='No. of steps',
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        '-t',
        '--target',
        dest='target',
        help='Target ImageNet class',
        type=str,
        required=False,
    )
    parser.add_argument(
        '-p',
        '--population',
        dest='population',
        help='Gene population',
        type=int,
        default=30,
        required=False,
    )
    parser.add_argument(
        '-d',
        '--dimensions',
        dest='dimensions',
        help='Image dimensions to create',
        type=int,
        default=[400, 400],
        required=False,
    )
    parser.add_argument(
        '-m',
        '--mutpb',
        dest='mutpb',
        help='Mutation probability',
        type=float,
        default=0.3,
        required=False,
    )
    parser.add_argument(
        '-I',
        '--indpb',
        dest='indpb',
        help='Independent probability for shuffles',
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        '-s',
        '--tournsize',
        dest='tournsize',
        help='Tournament selection size',
        type=int,
        default=4,
        required=False,
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    # Create a Trainer
    t = Trainer(
        X=options.input_str,
        population=options.population,
        dims=options.dimensions,
    )
    # Set colorscheme and dimensions
    if options.colorscheme is not None:
        t.set_colors(colorscheme=options.colorscheme)
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
    best_img = best.artist.draw_from_gene(best_gene)
    best_img.save(options.output_dir)


if __name__ == "__main__":
    main()
