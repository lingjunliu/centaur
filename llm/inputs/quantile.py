
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def quantile_inputs():
    list_of_inputs = []

    # Example 1: Basic example with float input and scalar q
    input_dict = {
        "input": torch.randn(2, 3).numpy(),
        "q": np.array(0.5, dtype=np.float32),
        "dim": 1,
        "keepdim": True,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Float input and 1D tensor q
    input_dict = {
        "input": torch.randn(4, 5).numpy(),
        "q": np.array([0.25, 0.5, 0.75], dtype=np.float32),
        "dim": 0,
        "keepdim": False,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Integer input and scalar q, different interpolation
    input_dict = {
        "input": torch.randint(0, 10, (3, 4)).float().numpy(),
        "q": np.array(0.6, dtype=np.float32),
        "dim": 1,
        "keepdim": True,
        "interpolation": "higher"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Negative values and 'nearest' interpolation
    input_dict = {
        "input": torch.randint(-5, 5, (2, 2)).float().numpy(),
        "q": np.array(0.3, dtype=np.float32),
        "dim": 1,
        "keepdim": False,
        "interpolation": "nearest"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Multiple dimensions, 'midpoint' interpolation, flatten
    input_dict = {
        "input": torch.randn(2, 3, 4).numpy(),
        "q": np.array(0.5, dtype=np.float32),
        "dim": None,
        "keepdim": False,
        "interpolation": "midpoint"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: scalar q, dim = None
    input_dict = {
        "input": torch.arange(10).float().numpy(),
        "q": np.array(0.7, dtype=np.float32),
        "dim": None,
        "keepdim": False,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: 1D Tensor Input and scalar q
    input_dict = {
        "input": torch.arange(5).float().numpy(),
        "q": np.array(0.4, dtype=np.float32),
        "dim": 0,
        "keepdim": False,
        "interpolation": "linear"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = quantile_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('quantile', generated_inputs)
