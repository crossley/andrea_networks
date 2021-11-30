# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:03:25 2021

@author: mq20185996
"""

from fastai.vision.all import *
import torch.nn as nn
import warnings
import random, textwrap
import os, glob
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import skimage
from skimage.util import random_noise
from skimage import util
from skimage import exposure
from scipy import ndimage
from datetime import datetime
import torch
import torch.utils.model_zoo
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from pathlib import Path
import collections
import seaborn as sn
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from SiameseNet import *
from SiameseNet0 import *
from SiameseNet1 import *
from SiameseNet12 import *
from SiameseNet2 import *
from SiameseNet22 import *

from util_train_nets import *
