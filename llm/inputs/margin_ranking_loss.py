
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def margin_ranking_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.5, 2.8], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.5,
        'size_average': True,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different margin and reduction mode
    input1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input2 = np.array([0.0, 0.5, 0.8], dtype=np.float32)
    target = np.array([-1, -1, 1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 1.0,
        'size_average': True,
        'reduce': False,
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multidimensional inputs
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    target = np.array([[1, -1], [-1, 1]], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.2,
        'size_average': True,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes
    input1 = np.array([1.0, 2.0], dtype=np.float32)
    input2 = np.array([2.0, 3.0], dtype=np.float32)
    target = np.array([1, -1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.3,
        'size_average': True,
        'reduce': False,
        'reduction': 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Zero margin
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([2.0, 2.5, 2.8], dtype=np.float32)
    target = np.array([1, -1, 1], dtype=np.int32)
    input_dict = {
        'input1': input1,
        'input2': input2,
        'target': target,
        'margin': 0.0,
        'size_average': True,
        'reduce': False,
        'reduction': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = margin_ranking_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('margin_ranking_loss', generated_inputs)
