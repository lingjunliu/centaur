
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def vector_to_parameters_inputs():
    list_of_inputs = []

    # Input 1: Basic test with float parameters
    parameters1 = [torch.randn(2, 3), torch.randn(4)]
    vec1 = torch.cat([p.view(-1) for p in parameters1])
    input_dict1 = {"vec": vec1.numpy(), "parameters": parameters1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Test with integer parameters
    parameters2 = [torch.randint(0, 10, (2, 2)), torch.randint(0, 5, (3,))]
    vec2 = torch.cat([p.view(-1).float() for p in parameters2])  # Convert to float for concatenation
    input_dict2 = {"vec": vec2.numpy(), "parameters": parameters2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Test with negative values and different shapes
    parameters3 = [torch.randn(1, 5, 5) * -1, torch.randn(2, 2, 2) * -0.5]
    vec3 = torch.cat([p.view(-1) for p in parameters3])
    input_dict3 = {"vec": vec3.numpy(), "parameters": parameters3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Test with single parameter tensor
    parameters4 = [torch.randn(10,)]
    vec4 = torch.cat([p.view(-1) for p in parameters4])
    input_dict4 = {"vec": vec4.numpy(), "parameters": parameters4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs = vector_to_parameters_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vector_to_parameters', generated_inputs)
