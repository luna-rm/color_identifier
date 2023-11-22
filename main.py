import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets, models
from collections import OrderedDict

import model_functions
import processing_functions

import json
import argparse

#python -m visdom.server

BATCH_SIZE = 64
DATA_DIR = "./images"
TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR = DATA_DIR + "/test"

CLASSES = ["blue", "green", "red"]


parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 10000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 20, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arguments = parser.parse_args()


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transforms = transforms.Compose([transforms.Resize(244), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform = transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle = True)

test_dataset = datasets.ImageFolder(TEST_DIR, transform = transforms)
test_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle = True)

    
if arguments.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif arguments.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
    
print(model)

# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, arguments.hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(arguments.hidden_units, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

# Loss function (since the output is LogSoftmax, we use NLLLoss)
criterion = nn.NLLLoss()

# Gradient descent optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
    
model_functions.train_classifier(model, optimizer, criterion, arguments.epochs, train_loader, test_loader, arguments.gpu)
    
model_functions.test_accuracy(model, test_loader, arguments.gpu)

model_functions.save_checkpoint(model, train_dataset, arguments.arch, arguments.epochs, arguments.learning_rate, arguments.hidden_units, input_size)  
