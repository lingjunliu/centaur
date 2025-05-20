
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def meshgrid_inputs():
    list_of_inputs = []

    # Input 1: Basic case with two 1D tensors (int) and ij indexing
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([4, 5, 6, 7], dtype=np.int64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two 1D tensors (float) and xy indexing
    x = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    y = np.array([5.5, 6.6, 7.7], dtype=np.float64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three 1D tensors (int, float, int) and ij indexing
    x = np.array([1, 2], dtype=np.float64)
    y = np.array([3.3, 4.4, 5.5], dtype=np.float64)
    z = np.array([6, 7, 8, 9], dtype=np.float64)
    input_dict = {
        "tensors": [x, y, z],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Two 1D tensors with negative values (int) and xy indexing
    x = np.array([-1, -2, -3], dtype=np.int64)
    y = np.array([-4, -5, -6, -7], dtype=np.int64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three 1D tensors with a scalar and ij indexing
    x = np.array([1], dtype=np.int64)
    y = np.array([2, 3, 4], dtype=np.int64)
    z = np.array([5, 6], dtype=np.int64)
    input_dict = {
        "tensors": [x, y, z],
        "indexing": 'ij'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Two 1D tensors with a large range and xy indexing
    x = np.linspace(-10, 10, 5, dtype=np.float64)
    y = np.linspace(-5, 5, 3, dtype=np.float64)
    input_dict = {
        "tensors": [x, y],
        "indexing": 'xy'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = meshgrid_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('meshgrid', generated_inputs)
