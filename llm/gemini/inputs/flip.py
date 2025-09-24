
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def flip_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    dims1 = (0,)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor
    input2 = torch.randint(0, 10, (3, 4)).numpy()
    dims2 = (0, 1)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex tensor
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    dims3 = (1, 2)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor with negative dimension
    input4 = torch.randn(1, 2, 3, 4).numpy()
    dims4 = (-1,)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5D tensor
    input5 = torch.randn(2, 2, 2, 2, 2).numpy()
    dims5 = (0, 2, 4)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.empty(0).numpy()
    dims6 = (0,)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Boolean tensor
    input7 = torch.randint(0, 2, (2, 3), dtype=torch.bool).numpy()
    dims7 = (0, 1)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = flip_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('flip', generated_inputs)
