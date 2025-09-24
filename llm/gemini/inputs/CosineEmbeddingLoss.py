
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def CosineEmbeddingLoss_inputs():
    generated_inputs = []

    input1 = np.random.randn(3, 5).astype(np.float32)
    input2 = np.random.randn(3, 5).astype(np.float32)
    target = np.array([1, -1, 1]).astype(np.int8)
    input_dict = {'margin': 0.0, 'size_average': None, 'reduce': None, 'reduction': 'mean', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(2, 4).astype(np.float64)
    input2 = np.random.randn(2, 4).astype(np.float64)
    target = np.array([-1, -1]).astype(np.int8)
    input_dict = {'margin': 0.5, 'size_average': True, 'reduce': True, 'reduction': 'sum', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(5, 2).astype(np.float32)
    input2 = np.random.randn(5, 2).astype(np.float32)
    target = np.array([1, -1, 1, -1, 1]).astype(np.int8)
    input_dict = {'margin': 0.2, 'size_average': False, 'reduce': True, 'reduction': 'mean', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(10, 1).astype(np.float64)
    input2 = np.random.randn(10, 1).astype(np.float64)
    target = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1]).astype(np.int8)
    input_dict = {'margin': 0.8, 'size_average': None, 'reduce': False, 'reduction': 'none', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))

    input1 = np.random.randn(1, 7).astype(np.float32)
    input2 = np.random.randn(1, 7).astype(np.float32)
    target = np.array([1]).astype(np.int8)
    input_dict = {'margin': 0.1, 'size_average': True, 'reduce': False, 'reduction': 'sum', 'input1':input1, 'input2':input2, 'target':target}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = CosineEmbeddingLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CosineEmbeddingLoss', generated_inputs)
