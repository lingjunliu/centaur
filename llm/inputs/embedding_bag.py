
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def embedding_bag_inputs():
    list_of_inputs = []

    # Example 1
    input1 = torch.tensor([1, 2, 4, 5, 4, 3, 0]).numpy()
    weight1 = torch.randn(10, 3).numpy()
    offsets1 = torch.tensor([0, 1, 2, 5]).numpy()
    input_dict1 = {
        "input": input1,
        "weight": weight1,
        "offsets": offsets1,
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2
    input2 = torch.tensor([0, 1, 2, 3]).numpy()
    weight2 = torch.randn(5, 4).numpy()
    offsets2 = torch.tensor([0, 2]).numpy()
    input_dict2 = {
        "input": input2,
        "weight": weight2,
        "offsets": offsets2,
        "max_norm": 1.0,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "mean",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3
    input3 = torch.tensor([1, 2, 3, 4, 0, 1]).numpy()
    weight3 = torch.randn(6, 2).numpy()
    offsets3 = torch.tensor([0, 3]).numpy()
    per_sample_weights3 = torch.randn(6).numpy()
    input_dict3 = {
        "input": input3,
        "weight": weight3,
        "offsets": offsets3,
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": True,
        "per_sample_weights": per_sample_weights3,
        "include_last_offset": False,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4
    input4 = torch.tensor([0, 1, 2, 0]).numpy()
    weight4 = torch.randn(4, 5).numpy()
    offsets4 = torch.tensor([0, 2, 4]).numpy()
    input_dict4 = {
        "input": input4,
        "weight": weight4,
        "offsets": offsets4,
        "max_norm": 2.0,
        "norm_type": 1,
        "scale_grad_by_freq": False,
        "mode": "max",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": False,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Example 5
    input5 = torch.tensor([0, 1, 2, 3, 0, 1, 2]).numpy()
    weight5 = torch.randn(7, 3).numpy()
    offsets5 = torch.tensor([0, 3, 5]).numpy()
    input_dict5 = {
        "input": input5,
        "weight": weight5,
        "offsets": offsets5,
        "max_norm": None,
        "norm_type": 2,
        "scale_grad_by_freq": False,
        "mode": "sum",
        "sparse": False,
        "per_sample_weights": None,
        "include_last_offset": True,
        "padding_idx": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = embedding_bag_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('embedding_bag', list_of_inputs)
