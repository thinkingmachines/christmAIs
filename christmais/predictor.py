# -*- coding: utf-8 -*-

"""Predict class and get the confidence of a target given an abstract art"""

# Import standard library
import json
import operator
import random
import logging

# Import modules
from gensim.test.utils import get_tmpfile
import matplotlib.pyplot as plt
import numpy as np
import requests

# Import from package
import torch
import torchvision
from torchvision import models
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO)
LABEL_SOURCE_URL = "https://s3.amazonaws.com/outcome-blog/imagenet/labels.json"
PRETRAINED_MODELS = {
    "resnet152": models.resnet152,
    "squeezenet1": models.squeezenet1_1,
    "resnet50": models.resnet50,
    "vgg16": models.vgg16,
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
                labels = json.load(f)
        except FileNotFoundError:
            msg = "File labels.json not found in /tmp/, attempting download from {}"
            logging.info(msg.format(LABEL_SOURCE_URL))
            r = requests.get(LABEL_SOURCE_URL, allow_redirects=True)
            with open(labels_file, "wb") as f:
                msg = "File labels.json stored in {}"
                logging.info(msg.format(labels_file))
                f.write(r.content)
            labels = json.load(f)
        return labels

    def predict(self, inp_file, target_label):
        """Calculates the score for each input image relative to the target label

        Parameters
        ----------
        inp_file: str
            Path to input image
        target_label: str
            An imagenet class label

        Returns
        -------
        dict
            Contains the class probabilities of each imagenet label for each model
	"""

        # TODO: Add error catching whenever KeyError/ValueError

        indices = [
            index
            for index, labels in self.labels.items()
            if target_label == labels[0]
        ]
        img_variable = self._preprocess(inp_file)

        scores, results = [], {}
        for model_name, model in self.models.items():
            probs = self.model_eval(model, img_variable)
            scores.append(probs[0][indices[0]])
            result = {
                label[0]: probs[0][index]
                for index, label in self.labels.items()
            }
            results[model_name] = result

        return np.mean(scores), results

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

    def _preprocess(self, img_pil):
        """Applies transforms to the input image

	    Parameters
	    ----------
	    img_file: PIL.Image
	        Path to input image

	    Returns
	    -------
	    torch.autograd.variable
	        Torch variable of the transformed image
	    """
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

        # img_pil = Image.open(img_file)
        img_tensor = preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        return img_variable


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
