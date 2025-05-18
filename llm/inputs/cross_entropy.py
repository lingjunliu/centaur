
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 10).numpy()
    target_tensor = torch.randint(0, 10, (2,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5, 3).numpy()
    target_tensor = torch.randint(0, 3, (5,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 7).numpy()
    target_tensor = torch.randint(0, 7, (1,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 2).numpy()
    target_tensor = torch.randint(0, 2, (4,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = cross_entropy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross_entropy', list_of_inputs)
