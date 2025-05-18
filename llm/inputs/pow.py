
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def pow_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.tensor(2.0)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 4).numpy()
    exponent_tensor = torch.tensor(0.5)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 5).numpy()
    exponent_tensor = torch.randn(1, 5, 5)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(3, 2, 1).numpy()
    exponent_tensor = torch.tensor(-1.0)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2).numpy()
    exponent_tensor = torch.tensor(2.0)
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = pow_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pow', list_of_inputs)
