
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def BCEWithLogitsLoss_inputs():
    list_of_inputs = []

    # Case 1: No optional arguments, provide dummy input and target
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: weight
    weight = np.random.rand(5).astype(np.float32)
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)

    input_dict = {"input": input_val, "target": target_val, "weight": weight}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: pos_weight
    pos_weight = np.random.rand(5).astype(np.float32)
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "pos_weight": pos_weight}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: reduction = 'sum'
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "reduction": "sum"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: reduction = 'none'
    input_val = np.random.rand(3, 5).astype(np.float32)
    target_val = np.random.randint(0, 2, size=(3, 5)).astype(np.float32)
    input_dict = {"input": input_val, "target": target_val, "reduction": "none"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = BCEWithLogitsLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('BCEWithLogitsLoss', generated_inputs)
