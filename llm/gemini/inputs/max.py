
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def generate_max_inputs():
    generated_inputs = []

    # Input 1: 2D float tensor, dim=1, keepdim=False
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {'input': input_tensor, 'dim': 1, 'keepdim': False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D int tensor with negative values, dim=0, keepdim=True
    input_tensor = torch.randint(-10, 10, (2, 3, 5)).int().numpy()
    input_dict = {'input': input_tensor, 'dim': 0, 'keepdim': True}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor, dim=0, keepdim=False
    input_tensor = torch.randn(5).numpy()
    input_dict = {'input': input_tensor, 'dim': 0, 'keepdim': False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor, dim=2, keepdim=True
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {'input': input_tensor, 'dim': 2, 'keepdim': True}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor with large values, dim=1, keepdim=False
    input_tensor = (torch.rand(2, 3) * 1000).numpy()
    input_dict = {'input': input_tensor, 'dim': 1, 'keepdim': False}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D tensor, dim=-1, keepdim=True (testing negative dimension)
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {'input': input_tensor, 'dim': -1, 'keepdim': True}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor with mixed positive and negative, dim=1, keepdim=False
    input_tensor = torch.randint(-5, 5, (2, 4, 3)).float().numpy()
    input_dict = {'input': input_tensor, 'dim': 1, 'keepdim': False}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = generate_max_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max', generated_inputs)
