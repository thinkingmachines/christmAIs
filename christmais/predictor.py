# -*- coding: utf-8 -*-

"""Predict class and get the confidence of a target given an abstract art"""

# Import standard library
import json
import logging
import operator
import random

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import requests
from gensim.test.utils import get_tmpfile

# Import from package
import torch
from torch.autograd import Variable
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO)
LABEL_SOURCE_URL = "https://s3.amazonaws.com/outcome-blog/imagenet/labels.json"
PRETRAINED_MODELS = {
    "resnet152": models.resnet152(pretrained=True),
    "squeezenet1": models.squeezenet1_1(pretrained=True),
    "resnet50": models.resnet50(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
}


class Predictor:
    """A class for mapping an input image into an ImageNet class

    The Predictor class uses a pretrained model of ResNet-152 for:
        - finding the closest ImageNet class for a given abstract art or
        - assigning a prediction confidence of an abstract art to a target
          ImageNet class
    """

    def __init__(
        self, models=["resnet152"], labels_file="labels.json", seed=42
    ):
        """Initialize the model

        Parameters
        ----------
        models : list of str (default is ["resnet152"])
            Define the models for prediction. Note that more models
            can severely affect prediction time.
        labels_file : str
            Filename where ImageNet labels are stored
        seed : int
            Random seed
        """
        self.logger = logging.getLogger(__name__)
        # Set random seed
        self._set_seed(seed)
        # Get labels
        self.labels = self._get_labels(labels_file)
        # Distributions
        self.distribs = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        # Get models
        self.models = self._get_models(models)

    def _set_seed(self, seed):
        """Set the random seeds for pytorch, numpy, and core python

        Parameters
        ----------
        seed : int
            Random seed
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _get_labels(self, labels_file):
        """Get labels from the source url

        Parameters
        ----------
        labels_file : str
            Filename where ImageNet labels are stored

        Returns
        -------
        dict
            Labels and numeric keys for ImageNet classes
        """
        labels_file = get_tmpfile(labels_file)

        try:
            with open(labels_file, "r") as f:
                msg = "Loading labels file {}"
                logging.info(msg.format(labels_file))
                labels = json.load(f)
                # Just typecast the keys
                labels = {int(k): v for k, v in labels.items()}
        except FileNotFoundError:
            msg = "File labels.json not found in /tmp/, attempting download from {}"
            logging.info(msg.format(LABEL_SOURCE_URL))
            r = requests.get(LABEL_SOURCE_URL).json().items()
            lbl = {int(k): v.split(", ") for k, v in r}
            with open(labels_file, "w") as f:
                msg = "File labels.json stored in {}"
                logging.info(msg.format(labels_file))
                json.dump(lbl, f, sort_keys=True, indent=4)
            labels = lbl
        return labels

    def _get_models(self, models):
        """Get pretrained models for chosen model

        Parameters
        ----------
        models : list of str (default is ["resnet152"])
            Define the models for prediction. Note that more models
            can severely affect prediction time.
        """
        try:
            mlist = dict((model, PRETRAINED_MODELS[model]) for model in models)
        except KeyError:
            msg = "Unrecognized model. Please choose among\
resnet152, vgg16, squeezenet1, resnet50"
            self.logger.exception(msg)
            raise
        else:
            return mlist

    def predict(self, X, target, top_classes=5):
        """Calculates the score for each input image relative to the target label

        Parameters
        ----------
        X : PIL.Image
            Input image matrix
        target: str
            The target ImageNet class label.
        top_classes : int (default is 5)
            Number of top classes to return

        Returns
        -------
        (float, dict)
            A tuple of values where float is the class probability of the
            target ImageClass, and dict contains the top_classes classes with
            the highest class probabilities
        """
        # Get label indices for the chosen target class
        indices = [idx for idx, lbl in self.labels.items() if target == lbl[0]]
        X = self._preprocess(X)
        print(indices)

        scores, results = [], {}
        for model_name, model in self.models.items():
            probs = self._model_eval(model, X)
            scores.append(probs[0][indices[0]])
            result = {
                label[0]: probs[0][index]
                for index, label in self.labels.items()
            }
            results[model_name] = result

        return np.mean(scores), results

    def _preprocess(self, X):
        """Transform the input image

        The preprocessing pipeline performs the following operations:
            - Resize an image of any given size into (224, 224)
            - Performs a CenterCrop given size 224
            - Typecasts the numpy.ndarray into tensor (ToTensor)
            - Normalizes the input image given self.distribs

        Parameters
        ----------
        X : PIL.Image
            Input image matrix

        Returns
        -------
        torch.autograd.variable
            Torch variable of the transformed image
        """
        # Create preprocessing pipeline
        # fmt: off
        preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.distribs["mean"],
                    self.distribs["std"])
                ])
        # fmt: on
        img_tensor = preprocess(X)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        return img_variable

    def _model_eval(self, model, img_variable):
        """Calculate the probabilities per imagenet class for a given image

        Parameters
        ----------
        model: torchvision.model
            A Pytorch CNN model pretrained on Imagenet

        Returns
        -------

	"""
        model.eval()
        fc_out = model(img_variable)
        sm = torch.nn.Softmax()
        probs = sm(fc_out)
        return probs.data.numpy()

    def plot_results(self, results, top_n=10, size=(3, 4)):
        """Plots the probabilities of the top n labels

	    Parameters
	    ----------
	    results : dict
	        Contains the class probabilities of each imagenet label for each model
	    top_n : int (default is 10)
	        The number of imagenet labels to plot, for each model
	    size: tuple (default is (3,4))
	        Size of each graph
	    """
        plt.figure(figsize=(size[0] * len(results), size[1]))
        for idx, model in enumerate(results):
            top_values = dict(
                sorted(
                    results[model].items(),
                    key=operator.itemgetter(1),
                    reverse=True,
                )[:top_n]
            )
            plt.subplot(1, len(results), idx + 1)
            plt.bar(
                range(len(top_values)),
                list(top_values.values()),
                align="center",
            )
            plt.xticks(
                range(len(top_values)), list(top_values.keys()), rotation=90
            )
            plt.title(model)
            plt.ylim(0, 1)
            plt.tight_layout()
        plt.show()
