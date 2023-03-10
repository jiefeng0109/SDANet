from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CTDetDataset

from .Dubai import Dubai
from .SanDiego import SanDiego
from .LasVegas import LasVegas

dataset_factory = {
    'Dubai': Dubai,
    'SanDiego': SanDiego,
    'LasVegas': LasVegas
}

_sample_factory = {
    'ctdet': CTDetDataset
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
