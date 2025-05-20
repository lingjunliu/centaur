
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def is_storage_inputs():
    list_of_inputs = []

    # Input 1: Float storage
    float_tensor = torch.randn(5, 5)
    float_storage = float_tensor.storage()
    input_dict = {"obj": float_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int storage
    int_tensor = torch.randint(0, 10, (3, 3), dtype=torch.int32)
    int_storage = int_tensor.storage()
    input_dict = {"obj": int_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Double Storage
    double_tensor = torch.randn(2, 2, dtype=torch.float64)
    double_storage = double_tensor.storage()
    input_dict = {"obj": double_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Byte Storage
    byte_tensor = torch.randint(0, 256, (4, 4), dtype=torch.uint8)
    byte_storage = byte_tensor.storage()
    input_dict = {"obj": byte_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bool Storage
    bool_tensor = torch.tensor([[True, False], [False, True]])
    bool_storage = bool_tensor.storage()
    input_dict = {"obj": bool_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty Storage
    empty_tensor = torch.empty(0)
    empty_storage = empty_tensor.storage()
    input_dict = {"obj": empty_storage}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = is_storage_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('is_storage', generated_inputs)
