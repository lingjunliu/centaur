
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(10).numpy()
    shape1 = (2, 5)
    input_dict1 = {"input": input1, "shape": shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor
    input2 = torch.randint(0, 10, (5, 4)).numpy()
    shape2 = (2, 2, 5)
    input_dict2 = {"input": input2, "shape": shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex tensor
    input3 = torch.randn(2, 3, 4, dtype=torch.complex64).numpy()
    shape3 = (3, 8)
    input_dict3 = {"input": input3, "shape": shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D bool tensor
    input4 = torch.randint(0, 2, (2, 2, 2, 2)).bool().numpy()
    shape4 = (4, 4)
    input_dict4 = {"input": input4, "shape": shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5D float tensor
    input5 = torch.randn(1, 2, 3, 4, 5).numpy()
    shape5 = (2, 3, 20)
    input_dict5 = {"input": input5, "shape": shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Reshape to -1 (infer size)
    input6 = torch.randn(2, 3, 4).numpy()
    shape6 = (-1,)
    input_dict6 = {"input": input6, "shape": shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Reshape from -1
    input7 = torch.randn(24).numpy()
    shape7 = (2, 3, -1)
    input_dict7 = {"input": input7, "shape": shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: empty tensor
    input8 = torch.empty(0).numpy()
    shape8 = (0,)
    input_dict8 = {"input": input8, "shape": shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs = reshape_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('reshape', generated_inputs)
