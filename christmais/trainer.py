# -*- coding: utf-8 -*-

"""Trainer system for integration"""

# Import standard library
import os
import operator
import logging

# Import modules
from deap import tools
import numpy as np
from tqdm import trange

from .drawsys import Artist
from .embedder import get_fasttext_pretrained
from .predictor import Predictor

logging.basicConfig(level=logging.INFO)


class Trainer:
    """Trainer class for integration"""

    def __init__(
        self,
        X,
        population=30,
        dims=(224, 224),
        seed=42,
        predictor_kwargs={},
        embedding_kwargs={},
    ):
        """Initialize the class

        Parameters
        ----------
        X : str
            Input sentence to transform into
        population : int (default is 30)
            Number of artists created
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
        self.X = X
        self.population = population
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

    def train(
        self,
        target,
        mutpb=0.3,
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

        # Compute for fitness
        fitness = self._fitness_fcn(imgs, target)
        genes = self._batch_get_genes()

        # Create initial population
        init_population = [
            Individual(img, gene, fitness, artist)
            for img, gene, fitness, artist in zip(
                imgs, genes, fitness, self.artists
            )
        ]

        # Save to filesystem
        if outdir is not None:
            dir_ = outdir + '/gen00/'
            for idx, indiv in enumerate(init_population):
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                indiv.image.save(dir_ + '{}_{}.png'.format(idx, indiv.fitness))

        # Start optimization via genetic algorithm
        population = init_population.copy()
        self.logger.info('Optimization has started')
        with trange(steps, desc='GEN', ncols=100, unit='gen') as t:
            for gen in t:

                # Filesystem IO
                if outdir is not None:
                    dir_ = outdir + '/gen{}/'.format(str(gen).zfill(2))

                next_pop = []  # Next population

                for idx in range(len(population)):
                    # Select parents using tournament selection
                    parents = tools.selTournament(
                        population, k=2, tournsize=4, fit_attr='fitness'
                    )
                    best_parent = max(
                        parents, key=operator.attrgetter('fitness')
                    )

                    # Generate new child using uniform crossover
                    c_artist = best_parent.artist
                    c_gene = tools.cxUniform(
                        parents[0].gene.copy(), parents[1].gene.copy(), indpb
                    )[0]
                    if np.random.uniform() < mutpb:
                        c_gene = tools.mutShuffleIndexes(c_gene, indpb)[0]
                    c_image = c_artist.draw_from_gene(c_gene)
                    c_fitness = self._fitness_fcn([c_image], target=target)[0]
                    child = Individual(c_image, c_gene, c_fitness, c_artist)

                    # Append child to next generation
                    next_pop.append(child)

                    # Filesystem IO
                    if outdir is not None:
                        if not os.path.exists(dir_):
                            os.makedirs(dir_)
                        child.image.save(
                            dir_ + '{}_{}.png'.format(idx, child.fitness)
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

        # Get best child and return it
        best_child = max(population, key=operator.attrgetter('fitness'))
        return best_child

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
