
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def vander_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default N and increasing
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"x": x, "N": None, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Specify N
    x = torch.tensor([1, 2, 3, 4]).numpy()
    input_dict = {"x": x, "N": 2, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specify increasing=True
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"x": x, "N": None, "increasing": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float tensor with N and increasing
    x = torch.tensor([1.5, 2.5, 3.5]).numpy()
    input_dict = {"x": x, "N": 4, "increasing": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    x = torch.tensor([-1, 0, 1]).numpy()
    input_dict = {"x": x, "N": None, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger N than len(x)
    x = torch.tensor([1, 2]).numpy()
    input_dict = {"x": x, "N": 5, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: N = 1
    x = torch.tensor([1, 2, 3]).numpy()
    input_dict = {"x": x, "N": 1, "increasing": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = vander_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vander', generated_inputs)
