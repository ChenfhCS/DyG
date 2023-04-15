import numpy as np
import pandas as pd
import torch
import torch.distributed
from torch._six import string_classes
from torch.utils.data import BatchSampler, Dataset, Sampler

from typing import Iterable, Iterator, List, Optional, Union

class DistributedBatchSampler(BatchSampler):
    """
    Distributed batch sampler.
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int, drop_last: bool,
                 rank: int, world_size: int,
                 num_chunks: int = 1):
        