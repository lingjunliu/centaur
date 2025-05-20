
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def vdot_inputs():
    list_of_inputs = []

    input1 = torch.tensor([2, 3]).numpy()
    input2 = torch.tensor([2, 1]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input2 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([-1, -2, -3]).numpy()
    input2 = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.tensor([1 + 2j, 3 - 1j]).numpy()
    input2 = torch.tensor([2 + 1j, 4 - 0j]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.tensor([2 + 1j, 4 - 0j]).numpy()
    input2 = torch.tensor([1 + 2j, 3 - 1j]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1, 2, 3], dtype=torch.float32).numpy()
    input2 = torch.tensor([4, 5, 6], dtype=torch.float32).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1, 2, 3], dtype=torch.int64).numpy()
    input2 = torch.tensor([4, 5, 6], dtype=torch.int64).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.tensor([1 + 1j, 2 - 2j, 3 + 0j]).numpy()
    input2 = torch.tensor([4 - 1j, 5 + 2j, 6 - 0j]).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = vdot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vdot', generated_inputs)
