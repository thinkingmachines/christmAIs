# -*- coding: utf-8 -*-

"""Drawing system to transform a numerical vector into an image

This system is based from Tom White's Perception Engine in
https://github.com/dribnet/dopes
"""

# Import standard library
import logging

# Import modules
import numpy as np
from PIL import Image, ImageDraw
from sklearn.preprocessing import minmax_scale

logging.basicConfig(level=logging.INFO)


class Artist:
    """Artist class for the drawing system

    For the Artist, the embedding vector corresponds to the computed
    word embeddings of the FastTextWrapper. These values serve as a seed to
    generate the images

    Usage
    -----

    Simply initialize the DrawingSystem with the embedding (an 8-element
    vector):

    .. code-block:: python

        import numpy as np
        from christmais.drawsys import Artist

        # Let's create a "random" embedding seed
        embedding = np.random.uniform(low=0, high=1, size=8)
        artist = Artist(embedding)

    This automatically computes the background colors and other required
    artifacts. In order to draw the resulting image, simply call:

    .. code-block:: python

        artist.draw()
    """

    def __init__(self, embedding, dims=(224, 224)):
        """Initialize the artist

        Parameters
        ----------
        embedding : numpy.ndarray
            Vector of size 8 for seeding the colors
        dims : tuple of size 2
            Dimensions of the resulting image
        """
        self.logger = logging.getLogger(__name__)
        self.emb = embedding
        self.dims = dims
        self.colors = self._generate_colors(self.emb)

    def draw(self, circle_density=10, line_density=10):
        """Draw the resulting image

        This is the main workhorse for the drawing system. Although the
        background colors were generated randomly, you can override them by
        setting the colors attribute with an RGB tuple

        Returns
        -------
        PIL.Image
            The resulting image
        """
        # Draw background
        im = Image.new("RGB", self.dims, self.colors["background"])
        draw = ImageDraw.Draw(im, "RGBA")
        # Draw circles
        circle_layers = {
            k: self.colors[k] for k in ("layer1", "layer2", "layer3")
        }
        im, draw = self._draw_circles(
            im=im, draw=draw, layers=circle_layers, density=circle_density
        )
        im, draw = self._draw_lines(
            im=im, draw=draw, color=self.colors["lines"], density=line_density
        )
        return im

    def _draw_circles(self, im, draw, layers, density=10):
        """Draw circles

        Parameters
        ----------
        im : PIL.Image
            Generated image
        draw : PIL.ImageDraw
            Drawable PIL object
        layers : dict with keys ["layer1", "layer2", "layer3"]
            Define the colors for each circle layers
        density : int (default is 10)
            Number of circles per layer

        Returns
        -------
        (PIL.Image, PIL.ImageDraw)
        """
        bottom_min_width = 0.02 * self.dims[0]
        bottom_max_width = 0.2 * self.dims[0]
        # Generate candidate coordinates
        cands = self._generate_coords(self.emb, nb_candidates=density)
        # Draw circles
        for i, (_, color) in enumerate(layers.items()):
            # Make each layer smaller than the one below
            min_width = bottom_min_width / (i + 1)
            max_width = bottom_max_width / (i + 1)
            for c in cands:
                w = self._interpolate(c[0], target=(min_width, max_width))
                # Randomly choose a coord from the candidate coords
                coords_ = np.random.choice(c, size=6)
                coords = self._interpolate(
                    coords_, target=(w, self.dims[0] - w)
                )
                x1, y1, x2, y2, x3, y3 = coords
                # Draw ellipses
                draw.ellipse([x1 - w, y1 - w, x1 + w, y1 + w], fill=color)
                draw.ellipse([x2 - w, y2 - w, x2 + w, y2 + w], fill=color)
                draw.ellipse([x3 - w, y3 - w, x3 + w, y3 + w], fill=color)
        return (im, draw)

    def _draw_lines(self, im, draw, color, density):
        """Draw lines

        Parameters
        ----------
        im : PIL.Image
            Generate image
        draw : PIL.ImageDraw
            Drawable PIL object
        color : tuple
            Line color
        density : int
            Amount of lines drawn
        """
        min_width = 0.004 * self.dims[0]
        max_width = 0.04 * self.dims[0]
        cands = self._generate_coords(self.emb, nb_candidates=density)
        for c in cands:
            w = self._interpolate(c[0], target=(min_width, max_width))
            width = 2 * w + 2
            # Randomly choose a coord from the candidate coords
            coords_ = np.random.choice(c, size=4)
            coords = self._interpolate(coords_, target=(w, self.dims[0] - w))
            x1, y1, x2, y2 = coords
            # Draw line with round line caps (circles at the end)
            draw.line((x1, y1, x2, y2), fill=color, width=width)
            draw.ellipse((x1 - w, y1 - w, x1 + w, y1 + w), fill=color)
            draw.ellipse((x2 - w, y2 - w, x2 + w, y2 + w), fill=color)
        return (im, draw)

    def _generate_coords(self, x, nb_candidates=10):
        """Sample candidate coordinates given a seed vector

        It generates candidate coordinates by sampling normally
        from the seed vector, treating it as the mean. Then,
        it is scaled via minmax scaling.

        Parameters
        ----------
        x : numpy.ndarray or float
            Seed vector, usually the embedding
        nb_candidates : int (default is 10)
            Number of candidates to generate

        Return
        ------
        numpy.ndarray
            Candidate coordinates sampled from the seed vector
            with shape (nb_candidates, x.shape[0])
        """
        cands = np.random.normal(loc=x, size=(nb_candidates, x.shape[0]))
        return minmax_scale(cands, feature_range=(0.02, 0.98))

    def _interpolate(self, x, current=(0, 1), target=(0, 255)):
        """Interpolate a given number into another range

        Parameters
        ----------
        x : numpy.ndarray or float
            Vector or value to interpolate
        current : tuple
            Current range
        target : tuple
            Target range

        Returns
        -------
        numpy.ndarray or int
            The value of vector x when interpolated to target range
        """
        y = np.interp(x, current, target)
        try:
            y = y.astype(int)
        except AttributeError:
            y = int(y)
        return y

    def _generate_colors(self, x):
        """Creates a color dictionary from the seed vector

        Parameters
        ----------
        x : numpy.ndarray
            Seed vector, usually the embedding

        Returns
        -------
        dict
            Color dictionary for drawing
        """
        layers = ["background", "layer1", "layer2", "layer3", "lines"]
        colors = dict.fromkeys(layers)
        for k, v in colors.items():
            # Choose three random elements from seed then
            # interpolate to range (0,255)
            c = self._interpolate(np.random.choice(x, size=4))
            colors[k] = tuple(c)
        self.logger.debug("Colors are now generated:\n {}".format(colors))
        return colors
