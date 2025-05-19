
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def NLLLoss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 1D input and target
    input = np.array([[-0.8, -0.2, -0.3]], dtype=np.float32)
    target = np.array([0], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D input and 1D target, different reduction
    input = np.array([[-0.8, -0.2, -0.3], [-0.1, -0.9, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: With weight
    input = np.array([[-0.8, -0.2, -0.3], [-0.1, -0.9, -0.5]], dtype=np.float32)
    target = np.array([0, 1], dtype=np.int64)
    weight = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    input_dict = {
        "input": input,
        "target": target,
        "weight": weight,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: With ignore_index
    input = np.array([[-0.8, -0.2, -0.3], [-0.1, -0.9, -0.5]], dtype=np.float32)
    target = np.array([0, -100], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different input shape
    input = np.array([[-0.8, -0.2], [-0.1, -0.9], [-0.5, -0.3]], dtype=np.float32)
    target = np.array([0, 1, 0], dtype=np.int64)
    input_dict = {
        "input": input,
        "target": target,
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = NLLLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('NLLLoss', generated_inputs)
