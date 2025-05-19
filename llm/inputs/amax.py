
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def amax_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = torch.randn(5).numpy()
    dim1 = (0,)
    keepdim1 = False
    input_dict1 = {"input": input1, "dim": dim1, "keepdim": keepdim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    dim2 = (0,)
    keepdim2 = True
    input_dict2 = {"input": input2, "dim": dim2, "keepdim": keepdim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 4: 4D float tensor with negative values
    input4 = torch.randn(2, 2, 2, 2).numpy() * -1
    dim4 = (1, 3)
    keepdim4 = True
    input_dict4 = {"input": input4, "dim": dim4, "keepdim": keepdim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D int tensor, no dim
    input5 = torch.randint(0, 10, (3, 4)).numpy()
    dim5 = None
    keepdim5 = False
    input_dict5 = {"input": input5, "dim": dim5, "keepdim": keepdim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D float tensor, dim is an int
    input6 = torch.randn(3, 4).numpy()
    dim6 = (1,)
    keepdim6 = False
    input_dict6 = {"input": input6, "dim": dim6, "keepdim": keepdim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 3D float tensor
    input7 = torch.randn(2, 3, 4).numpy()
    dim7 = (0, 2)
    keepdim7 = True
    input_dict7 = {"input": input7, "dim": dim7, "keepdim": keepdim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = amax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('amax', generated_inputs)
