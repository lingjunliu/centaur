
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def torch_histogram_inputs():
    list_of_inputs = []

    # Example 1: Basic usage with integer bins and default range
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": None,
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Specifying range
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": (0.0, 5.0),
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Using weights
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    weight_tensor = torch.tensor([1.0, 2.0, 1.5, 1.0, 0.5, 0.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": (0.0, 5.0),
        "weight": weight_tensor,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Using density
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": (0.0, 5.0),
        "weight": None,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Using tensor as bins
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0]).numpy()
    bins_tensor = torch.tensor([0.0, 1.5, 2.5, 3.5, 4.5, 5.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": bins_tensor,
        "range": None,
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Input tensor with negative values
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0]).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 4,
        "range": (-2.0, 3.0),
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Input tensor with float64
    input_tensor = torch.tensor([1.0, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=torch.float64).numpy()
    input_dict = {
        "input": input_tensor,
        "bins": 5,
        "range": None,
        "weight": None,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = torch_histogram_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('histogram', generated_inputs)
