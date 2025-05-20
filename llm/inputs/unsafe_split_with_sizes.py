
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unsafe_split_with_sizes_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with integer tensor and positive split sizes
    tensor = torch.arange(10).numpy()
    split_sizes = [2, 3, 5]
    dim = 0
    input_dict = {"tensor": tensor, "split_sizes": split_sizes, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = unsafe_split_with_sizes_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unsafe_split_with_sizes', generated_inputs)
