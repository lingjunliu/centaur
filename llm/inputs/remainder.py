
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def remainder_inputs():
    list_of_inputs = []

    input1 = torch.tensor([-3., -2, -1, 1, 2, 3]).numpy()
    other1 = torch.tensor(2).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    other2 = torch.tensor(-1.5).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (2, 3)).float().numpy()
    other3 = torch.tensor(2.5).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(-10, 10, (3, 2, 4)).int().numpy()
    other4 = torch.tensor(-3).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 4).double().numpy()
    other5 = torch.tensor(1.2).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.tensor([[-1, 2], [-3, 4]]).float().numpy()
    other6 = torch.tensor([[5, -6], [7, -8]]).float().numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([5]).int().numpy()
    other7 = torch.tensor(2).int().numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = remainder_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('remainder', generated_inputs)
