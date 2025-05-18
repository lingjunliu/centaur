
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def amax_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (0,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = (1, 2)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 7, 2).numpy()
    dim = (2,)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(3, 2).numpy()
    dim = (0, 1)
    keepdim = True
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 4, 4, 4).numpy()
    dim = (1, 3)
    keepdim = False
    input_dict = {"input": input_tensor, "dim": dim, "keepdim": keepdim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = amax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('amax', list_of_inputs)
