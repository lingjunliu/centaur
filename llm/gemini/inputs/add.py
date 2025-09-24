
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def add_inputs():
    list_of_inputs = []

    # Case 1: Basic addition with float tensors and alpha=1
    input1 = torch.randn(4).numpy()
    other1 = torch.randn(4).numpy()
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": other1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Addition with different shapes (broadcasting) and alpha=2
    input2 = torch.randn(4, 1).numpy()
    other2 = torch.randn(4).numpy()
    alpha2 = 2.0
    input_dict2 = {"input": input2, "other": other2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Addition with integer tensors and negative alpha
    input3 = torch.randint(-5, 5, (3, 3)).numpy()
    other3 = torch.randint(-5, 5, (3, 3)).numpy()
    alpha3 = -1
    input_dict3 = {"input": input3, "other": other3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: Addition with a scalar (number) and complex input
    input4 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    other4 = 3.0
    alpha4 = 1.0
    input_dict4 = {"input": input4, "other": other4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Addition with high dimensional tensor and alpha=0
    input5 = torch.randn(2, 3, 4, 5).numpy()
    other5 = torch.randn(2, 3, 4, 5).numpy()
    alpha5 = 0.0
    input_dict5 = {"input": input5, "other": other5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.add"] = add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('add', generated_inputs['torch.add'], lib="torch")
