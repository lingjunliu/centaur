
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def vsplit_inputs():
    list_of_inputs = []

    # Input 1: Splitting into equal sections
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Splitting at specified indices
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor splitting
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Uneven splitting with zero-sized tensor
    t = torch.arange(16.0).reshape(4, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [3, 6]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float tensor
    t = torch.randn(6, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer Tensor
    t = torch.randint(0, 10, (5, 5)).numpy()
    input_dict = {"input": t, "indices_or_sections": [2, 4]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D tensor with single split
    t = torch.arange(9.0).reshape(3, 3).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = vsplit_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vsplit', generated_inputs)
