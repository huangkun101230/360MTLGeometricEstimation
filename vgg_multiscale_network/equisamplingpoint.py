import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .GridGenerator import GridGenerator


def genSamplingPattern(h, w, kh, kw, stride=1):
    gridGenerator = GridGenerator(h, w, (kh, kw), stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    grid = LonLatSamplingPattern#np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      grid = torch.FloatTensor(grid)
      grid.requires_grad = False

    return grid