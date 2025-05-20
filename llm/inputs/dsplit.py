
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def dsplit_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D tensor with integer sections
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D tensor with integer sections that evenly divide
    t = torch.arange(16.0).reshape(2, 2, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D tensor with list of indices
    t = torch.arange(48.0).reshape(2, 2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor with float values and tuple of indices
    t = torch.randn(2, 3, 5).numpy()
    input_dict = {"input": t, "indices_or_sections": (2, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor with a split resulting in an empty tensor
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [5]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor with complex numbers
    t = torch.randn(2, 2, 4, dtype=torch.complex64).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D tensor
    t = torch.arange(32.0).reshape(1, 2, 2, 2, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor, uneven split using list
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1,2]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = dsplit_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('dsplit', generated_inputs)
