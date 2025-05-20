
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def is_same_size_inputs():
    list_of_inputs = []

    # Test case 1: Two tensors with the same size (1D)
    tensor1 = torch.randn(5).numpy()
    tensor2 = torch.randn(5).numpy()
    input_dict = {"tensor1": tensor1, "tensor2": tensor2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = is_same_size_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('is_same_size', generated_inputs)
