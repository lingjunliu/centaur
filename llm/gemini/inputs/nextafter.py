
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other2 = torch.tensor([2.0, 1.0, 4.0]).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([-1.0, -2.0, -3.0]).numpy()
    other3 = torch.tensor([-2.0, -1.0, -4.0]).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.zeros(2, 2).numpy()
    other4 = torch.ones(2, 2).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.ones(1, 5).numpy()
    other5 = (torch.ones(5) * 2).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1.0, 2.0]).float().numpy()
    other6 = torch.tensor([2.0, 1.0]).float().numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input8 = torch.randn(5).numpy()
    other8 = (torch.randn(1) * 10).numpy()
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs = nextafter_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nextafter', generated_inputs)
