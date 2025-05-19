
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []

    # Case 1: Basic case with 2D input and 1D target (long)
    input_tensor = torch.randn(3, 5).numpy()
    target_tensor = torch.randint(0, 5, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different batch size and number of classes
    input_tensor = torch.randn(5, 10).numpy()
    target_tensor = torch.randint(0, 10, (5,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Input with log probabilities (softmax already applied)
    input_tensor = torch.randn(4, 3).log_softmax(dim=1).numpy()
    target_tensor = torch.randint(0, 3, (4,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Larger tensors
    input_tensor = torch.randn(10, 20).numpy()
    target_tensor = torch.randint(0, 20, (10,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values in input (valid after softmax/log_softmax)
    input_tensor = torch.randn(3, 4).numpy()
    target_tensor = torch.randint(0, 4, (3,)).numpy()
    input_dict = {"input": input_tensor, "target": target_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = cross_entropy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross_entropy', generated_inputs)
