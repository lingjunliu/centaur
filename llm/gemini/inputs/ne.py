
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ne_inputs():
    list_of_inputs = []

    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other1 = torch.tensor([[1, 1], [4, 4]]).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(3, 4).numpy()
    other2 = 0.5
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (2, 2, 2)).numpy()
    other3 = torch.ones(2, 2, 2).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5).numpy()
    other4 = torch.randn(5).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randint(0, 10, (1, 5)).numpy()
    other5 = 5
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3).numpy()
    other6 = torch.full((3, 3), 5.0).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([[-1, -2], [-3, -4]], dtype=torch.int64).numpy()
    other7 = torch.tensor([0, -2], dtype=torch.int64).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = ne_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ne', generated_inputs)
