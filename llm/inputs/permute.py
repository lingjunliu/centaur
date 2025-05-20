
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def permute_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 5).numpy()
    dims1 = (2, 0, 1)
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (4, 2, 6, 3)).numpy()
    dims2 = (3, 1, 0, 2)
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5, 7, 2, 4).numpy()
    dims3 = (0, 4, 2, 3, 1)
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 4).numpy()
    dims4 = (1, 0)
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(10).numpy()
    dims5 = (0,)
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 2, 2).numpy()
    dims6 = (0, 1, 2)
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 3, 4, 5).numpy()
    dims7 = (3, 2, 1, 0)
    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(5,).numpy()
    dims8 = (0,)
    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = permute_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('permute', generated_inputs)
