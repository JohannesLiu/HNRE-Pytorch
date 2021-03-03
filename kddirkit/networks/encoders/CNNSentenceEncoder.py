import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim
from kddirkit.networks.embedders import WordVec
from kddirkit.networks.encoders import SentenceEncoder

