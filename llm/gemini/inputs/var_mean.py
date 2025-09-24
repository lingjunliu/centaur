
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, dim=None, correction=1, keepdim=False
    input_tensor = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": None,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor, dim=0, correction=0, keepdim=True
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, dim=1, correction=2, keepdim=False
    input_tensor = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "correction": 2,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor, dim=0, correction=1, keepdim=True
    input_tensor = np.random.randn(10).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5:  tensor with negative values, dim=(0, 2), correction=1, keepdim=True
    input_tensor = np.random.randn(2, 3, 4).astype(np.float32) - 0.5
    input_dict = {
        "input": input_tensor,
        "dim": (0, 2),
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor, dim=(1, 2), correction=1, keepdim=False
    input_tensor = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": (1, 2),
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Empty tensor, dim=None, correction=1, keepdim=False
    input_tensor = np.array([]).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "dim": None,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = var_mean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('var_mean', generated_inputs)
