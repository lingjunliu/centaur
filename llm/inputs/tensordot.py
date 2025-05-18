
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def tensordot_inputs():
    list_of_inputs = []

    a = torch.randn(3, 4, 5).numpy()
    b = torch.randn(4, 3, 2).numpy()
    dims = ([1, 0], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 3).numpy()
    b = torch.randn(3, 4).numpy()
    dims = (([1], [0]))
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(3, 4, 5).numpy()
    b = torch.randn(3, 5, 2).numpy()
    dims = ([0, 2], [0, 1])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 3, 4).numpy()
    b = torch.randn(4, 5).numpy()
    dims = ([2], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = torch.randn(2, 2).numpy()
    b = torch.randn(2).numpy()
    dims = ([1], [0])
    input_dict = {"a": a, "b": b, "dims": dims}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = tensordot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('tensordot', list_of_inputs)
