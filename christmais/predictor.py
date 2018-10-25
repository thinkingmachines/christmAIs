import io
import requests
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import random
import operator

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
LABELS = {int(key): value.split(", ") for (key, value) in requests.get(LABELS_URL).json().items()}

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

models = {
    'resnet152': torchvision.models.resnet152(pretrained=True),
    'squeezenet1': torchvision.models.squeezenet1_1(pretrained=True),
    'resnet50': torchvision.models.resnet50(pretrained=True),
    'vgg16': torchvision.models.vgg16(pretrained=True)
}


class Predictor: 

	def model_eval(self, model, img_variable):    
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

	def preprocess(self,  img_file):   
	    """Applies transforms to the input image 
	        
	    Parameters
	    ----------
	    img_file: str
	        Path to input image 

	    Returns
	    -------
	    torch.autograd.variable
	        Torch variable of the transformed image
	    """
	    normalize = torchvision.transforms.Normalize(
	       mean=IMGNET_MEAN,
	       std=IMGNET_STD
	    )
	    preprocess = torchvision.transforms.Compose([
	       torchvision.transforms.Resize(224),
	       torchvision.transforms.CenterCrop(224),
	       torchvision.transforms.ToTensor(),
	       normalize
	    ])
	    
	    img_pil = Image.open(img_file)
	    img_tensor = preprocess(img_pil)
	    img_tensor.unsqueeze_(0)
	    img_variable = Variable(img_tensor)
	    return img_variable

	def get_score(self, self, inp_file, target_label):
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
	    indices = [index for index, labels in LABELS.items() if target_label == labels[0]]
	    img_variable = preprocess(inp_file)
	    
	    scores, results = [], {}
	    for model_name, model in models.items():
	        probs = model_eval(model, img_variable)
	        scores.append(probs[0][indices[0]])
	        result = {label[0]: probs[0][index] for index, label in LABELS.items()}
	        results[model_name] = result
	    
	    return np.mean(scores), results

	def plot_results(self, results, top_n=10, size=(3,4)): 
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
	    fig = plt.figure(figsize=(size[0]*len(results), size[1]))
	    for idx, model in enumerate(results):
	        top_values = dict(sorted(results[model].items(), key=operator.itemgetter(1), reverse=True)[:top_n])
	        plt.subplot(1, len(results), idx+1)
	        plt.bar(range(len(top_values)), list(top_values.values()), align='center')
	        plt.xticks(range(len(top_values)), list(top_values.keys()), rotation=90)
	        plt.title(model)
	        plt.ylim(0,1)
	        plt.tight_layout()
	    plt.show()