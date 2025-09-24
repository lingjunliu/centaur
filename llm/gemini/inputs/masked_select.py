
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def masked_select_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor and boolean mask
    input_tensor = torch.randn(3, 4).numpy()
    mask = (torch.randn(3, 4) > 0).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor and boolean mask with different shape
    input_tensor = torch.randint(-5, 5, (2, 2, 2)).numpy()
    mask = (torch.randn(2, 2, 2) > 0.5).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor and boolean mask
    input_tensor = torch.complex(torch.randn(2, 3), torch.randn(2, 3)).numpy()
    mask = (torch.randn(2, 3) > 0).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor and mask
    input_tensor = torch.arange(10).float().numpy()
    mask = (torch.arange(10) % 2 == 0).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with negative values and boolean mask
    input_tensor = torch.randn(4, 4) * 10 - 5
    mask = (input_tensor > -2).numpy()
    input_dict = {"input": input_tensor, "mask": mask}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = masked_select_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('masked_select', generated_inputs)
