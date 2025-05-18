
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def float_power_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy()
    exponent1 = torch.tensor(2.0).numpy()
    input_dict1 = {"input": input1, "exponent": exponent1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(4, 4).numpy()
    exponent2 = torch.randn(4, 4).numpy()
    input_dict2 = {"input": input2, "exponent": exponent2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5, 5).numpy()
    exponent3 = torch.tensor(0.5).numpy()
    input_dict3 = {"input": input3, "exponent": exponent3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0, 2.0, 3.0])
    exponent4 = np.array([2.0, 3.0, 4.0])
    input_dict4 = {"input": input4, "exponent": exponent4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.ones(2, 2).numpy()
    exponent5 = torch.full((2, 2), 3.5).numpy()
    input_dict5 = {"input": input5, "exponent": exponent5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = float_power_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('float_power', list_of_inputs)
