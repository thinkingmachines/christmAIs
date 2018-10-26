# Import standard library
import io
import operator
import random

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

# Import from package
import torch
import torchvision
from torch.autograd import Variable

SEED = 42

# init param
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Predictor:
    def __init__(self, seed):

        # TODO: If labels is not None, do not download
        # def _get_labels()
        self.labels_url = (
            "https://s3.amazonaws.com/outcome-blog/imagenet/labels.json"
        )
        self.labels = {
            int(key): value.split(", ")
            for (key, value) in requests.get(self.labels_url).json().items()
        }

        self.models = {
            "resnet152": torchvision.models.resnet152(pretrained=True),
            "squeezenet1": torchvision.models.squeezenet1_1(pretrained=True),
            "resnet50": torchvision.models.resnet50(pretrained=True),
            "vgg16": torchvision.models.vgg16(pretrained=True),
        }

    def _model_eval(self, model, img_variable):
        """Calculates the probabilities per imagenet class for a given image

	    Parameters
	    ----------
	    model:
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

    def get_score(self, inp_file, target_label):
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
        fig = plt.figure(figsize=(size[0] * len(results), size[1]))
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
