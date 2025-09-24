
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def count_nonzero_inputs():
    list_of_inputs = []

    input_np = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([
        [1.0, 0.0, 2.5],
        [0.0, -3.2, 0.0]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([
        1, 2, 3, 4, 5
    ])
    input_dict = {"input": input_np, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4]
    ])
    input_dict = {"input": input_np, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4]
    ])
    input_dict = {"input": input_np, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_np = np.array([
        [[1, 0], [0, 2]],
        [[3, 0], [0, 4]]
    ])
    input_dict = {"input": input_np, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = count_nonzero_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('count_nonzero', generated_inputs)
