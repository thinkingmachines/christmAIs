# -*- coding: utf-8 -*-

"""Trainer system for integration"""

# Import standard library
import copy
import gc
import logging
import operator
import os
import random

# Import modules
import coloredlogs
import numpy as np
from tqdm import trange

# Import from package
from deap import tools

from .drawsys import Artist
from .embedder import get_fasttext_pretrained
from .predictor import Predictor

logging.basicConfig(level=logging.INFO)


class Trainer:
    """Trainer class for integration"""

    def __init__(
        self,
        X,
        colorscheme=None,
        population=100,
        dims=(224, 224),
        pool_size=30,
        seed=42,
        predictor_kwargs={},
        embedding_kwargs={},
    ):
        """Initialize the class

        Parameters
        ----------
        X : str
            Input sentence to transform into
        colorscheme : dict (default is None)
            Useful for setting the color scheme before hand.
            Dictionary with keys 'background', 'layer1', 'layer2', 'layer3',
            and 'lines'
        population : int (default is 30)
            Initial population, number of artists created
        pool_size : int (default is 30)
            Size of the initial population as the size of each generation in the
            gene pool as it evolves.
        dims : tuple of size 2
            Dimensions of the resulting image
        seed : int (default is 42)
            Sets the random seed
        predictor_kwargs : dict
            Arguments for Predictor() class
        embedding_kwargs : dict
            Arguments for FastTextWrapper() class
        """
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(logging.INFO, logger=self.logger)
        self.X = X
        self.population = population
        self.pool_size = pool_size
        self.dims = dims
        self.seed = seed
        self.predictor_kwargs = predictor_kwargs
        self.embedding_kwargs = embedding_kwargs
        # Auxiliary attributes
        self._bar_fmt = '{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}'
        # Create secondary attributes
        self.reset()

    def reset(self):
        """Generate a new set of artists, predictor, and model"""
        # Build a FastText model
        self.model = get_fasttext_pretrained(
            load=True, **self.embedding_kwargs
        )
        self.logger.info('Built FastText model')
        # Transform the string using the FastText model
        self.emb = self.model.transform(self.X)
        self.logger.info('Embedding created for `{}`'.format(self.X))
        # Generate a list of artists
        self.artists = [
            Artist(self.emb, self.dims) for i in range(self.population)
        ]
        self.logger.info('Created {} artists'.format(len(self.artists)))
        # Create a predictor
        self.predictor = Predictor(seed=self.seed, **self.predictor_kwargs)
        self.logger.info(
            'Initialized Predictor with {}'.format(
                self.predictor.models.keys()
            )
        )
        # Setting history
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def set_colors(self, colorscheme):
        """Set the artist colorscheme

        It is preferable to call this before training, so that the color
        dictionary is updated right away.

        Parameters
        ----------
        colorscheme : dict (default is None)
            Useful for setting the color scheme before hand.
            Dictionary with keys 'background', 'layer1', 'layer2', 'layer3',
            and 'lines'
        """
        self.logger.info('Setting colorscheme for artists')
        for k, v in colorscheme.items():
            assert (
                len(v) == 4
            ), 'Color values should be a tuple of size 4 (RGBA)'
        for artist in self.artists:
            artist.colors = colorscheme

    def set_dims(self, dims=(500, 500)):
        """Set the artist canvas dimensions

        It is preferable to call this before training, so that the dimensions
        are updated right away.

        Parameters
        ---------
        dims : tuple (default is (500, 500))
        """
        self.logger.info('Setting artist dimensions to {}'.format(dims))
        for artist in self.artists:
            artist.dims = dims

    def train(
        self,
        target,
        mutpb=0.05,
        indpb=0.5,
        k=2,
        tournsize=4,
        steps=100,
        outdir=None,
    ):
        """Train and generate images using a genetic algorithm

        Parameters
        ----------
        target : str
            The target ImageNet class label.
        mutpb : float (default is 0.3)
            Mutation probability
        indpb : float (default is 0.5)
            Independent probability for each attribute to be exchanged/shuffled
            during uniform crossover and/or mutation.
        k : int (default is 2)
            Number of individuals to select during tournament selection
        tournsize : int (default is 4)
            Number of individuals participating in each tournament
        steps : int (default is 100)
            The number of steps to run the optimization algorithm
        outdir : str (default is None)
            Output directory to where the images for each generation will be
            saved

        Returns
        -------
        christmais.trainer.Individual
            The best child during the optimization run.
        """

        self.logger.info('Initializing population and histories...')

        # Get initial images
        imgs = self._batch_draw()

        # Get all gene elements
        genes = self._batch_get_genes()
        all_genes = []
        for gene in genes:
            for i in range(len(gene)):
                all_genes.append(gene[i, :])

        # Compute fitness for each individual
        fitness = []
        imgs = []
        for gene in genes:
            c_image = self.artists[0].draw_from_gene(gene[:1, :])
            c_fitness = self._fitness_fcn([c_image], target=target)[0]
            fitness.append(c_fitness)
            imgs.append(c_image)

        # Create initial population
        init_population = [
            Individual(img, gene[:1, :], fitness, artist)
            for img, gene, fitness, artist in zip(
                imgs, genes, fitness, self.artists
            )
        ]

        # Save to filesystem
        if outdir is not None:
            self.logger.info('Writing image files at {}'.format(outdir))
            dir_ = outdir + '/gen00/'
            for idx, indiv in enumerate(init_population):
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                indiv.image.save(
                    dir_ + '{}_{}.png'.format(str(idx).zfill(2), indiv.fitness)
                )

        # Start optimization via genetic algorithm
        population = init_population.copy()
        self.logger.info('Optimization has started')
        with trange(steps, desc='GEN', ncols=100, unit='gen') as t:
            for gen in t:
                # Filesystem IO
                if outdir is not None:
                    dir_ = outdir + '/gen{}/'.format(str(gen + 1).zfill(2))

                next_pop = []  # Next population

                for idx in range(self.pool_size):
                    individuals = tools.selTournament(
                        population,
                        k=k,
                        tournsize=tournsize,
                        fit_attr='fitness',
                    )

                    tournament = []
                    for individual in individuals:
                        random_gene = random.choice(all_genes)
                        artist = individual.artist
                        gene = np.vstack((individual.gene, random_gene))
                        image = individual.artist.draw_from_gene(
                            individual.gene
                        )
                        fitness = self._fitness_fcn([image], target=target)[0]
                        tournament.append(
                            Individual(image, gene, fitness, artist)
                        )

                    best = max(tournament, key=operator.attrgetter('fitness'))

                    # Randomly remove elementes
                    best.gene = self._mut_delete(best.gene, mutpb=mutpb)
                    best.image = best.artist.draw_from_gene(best.gene)
                    best.fitness = self._fitness_fcn(
                        [best.image], target=target
                    )[0]

                    # Append child to next generation
                    next_pop.append(best)

                    # Filesystem IO
                    if outdir is not None:
                        if not os.path.exists(dir_):
                            os.makedirs(dir_)
                        best.image.save(
                            dir_
                            + '{}_{}.png'.format(
                                str(idx).zfill(2), best.fitness
                            )
                        )

                # Set new population
                population = next_pop

                # Get fitness
                best_fitness = max(
                    population, key=operator.attrgetter('fitness')
                ).fitness
                avg_fitness = np.mean([indiv.fitness for indiv in population])
                t.set_postfix({'best': best_fitness, 'avg': avg_fitness})

                # Save to history
                self.best_fitness_history.append(best_fitness)
                self.avg_fitness_history.append(avg_fitness)

                gc.collect()

        # Get best child and return it
        best_child = max(population, key=operator.attrgetter('fitness'))
        return best_child

    def _mut_delete(self, gene, mutpb):
        """Creates a mutation by deleting some genes

        Parameters
        ----------
        gene : np.ndarray
            Gene representation of an image
        mutpb : float
            Mutation parameter
        """
        for element in gene:
            if np.random.uniform() < mutpb:
                gene = np.delete(gene, element, 0)
        return gene

    def _batch_draw(self, genes=None):
        """Draw images from artists

        If genes are supplied, then it uses draw as reference

        Returns
        -------
        list of PIL.Image
            drawn images from Artists
        """
        if genes is not None:
            self.logger.debug('Using genes as reference')
            imgs = [a.draw_from_gene(g) for a, g in zip(self.artists, genes)]
        else:
            imgs = [a.draw() for a in self.artists]
        return imgs

    def _batch_get_genes(self, ravel=False):
        """Obtain the genes from all generated artists

        Parameters
        ----------
        ravel : bool
            Returns a gene of shape (population, 10*26)


        Returns
        -------
        np.ndarray
            Usually of shape (population, 10, 26)
        """
        if ravel:
            genes = [a.get_gene().ravel() for a in self.artists]
        else:
            genes = [a.get_gene() for a in self.artists]
        return np.stack(genes, axis=0)

    def _fitness_fcn(self, imgs, target):
        """Compute the fitness of the images given a target class label

        Parameter
        ---------
        imgs : list of PIL.Image
            A list of generated images for each Artist
        target : str
            The target ImageNet class label.

        Returns
        -------
        np.ndarray
            An array where each element is the fitness of each Artist
        """
        fitness = [self.predictor.predict(img, target)[0] for img in imgs]
        return np.asarray(fitness)


class Individual:
    """Individual class for DEAP integration

    Attributes
    ----------
    image : PIL.Image
    gene : np.ndarray
    fitness : float
    artist : christmais.Artist
    """

    def __init__(self, image, gene, fitness, artist):
        self.image = image
        self.gene = gene
        self.fitness = fitness
        self.artist = artist
