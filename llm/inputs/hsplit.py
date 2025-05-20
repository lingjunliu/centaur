
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def hsplit_inputs():
    list_of_inputs = []

    # Case 1: 2D tensor, integer sections
    t1 = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict1 = {"input": t1, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D tensor, list of indices
    t2 = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict2 = {"input": t2, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 1D tensor, integer sections
    t3 = torch.arange(12.0).numpy()
    input_dict3 = {"input": t3, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 1D tensor, list of indices
    t4 = torch.arange(12.0).numpy()
    input_dict4 = {"input": t4, "indices_or_sections": [2, 5, 8]}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 3D tensor, integer sections
    t5 = torch.arange(24.0).reshape(2, 4, 3).numpy()
    input_dict5 = {"input": t5, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: 3D tensor, list of indices
    t6 = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict6 = {"input": t6, "indices_or_sections": [1, 2]}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Case 7: 2D int tensor
    t7 = torch.arange(16).reshape(4, 4).numpy()
    input_dict7 = {"input": t7, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = hsplit_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hsplit', generated_inputs)
