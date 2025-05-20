
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def parameters_to_vector_inputs():
    list_of_inputs = []

    # Input 1: List of float tensors with different shapes
    params1 = [torch.randn(2, 3).numpy(), torch.randn(5).numpy(), torch.randn(1, 1, 4).numpy()]
    input_dict1 = {"parameters": params1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: List of int tensors with different shapes
    params2 = [torch.randint(0, 10, (2, 2)).numpy(), torch.randint(-5, 5, (3,)).numpy()]
    input_dict2 = {"parameters": params2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: List of mixed float and int tensors
    params3 = [torch.randn(3, 1).numpy(), torch.randint(0, 5, (2,)).numpy(), torch.randn(1).numpy()]
    input_dict3 = {"parameters": params3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: List containing a single tensor
    params4 = [torch.randn(4, 4).numpy()]
    input_dict4 = {"parameters": params4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: List with empty tensors
    params5 = [torch.randn(0).numpy(), torch.randn(2, 0).numpy(), torch.randn(0, 0, 0).numpy()]
    input_dict5 = {"parameters": params5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: List of tensors with negative values
    params6 = [torch.randn(2, 3) * -1.0, torch.randn(5) * -2.0, torch.randn(1, 1, 4) * -0.5]
    input_dict6 = {"parameters": [p.numpy() for p in params6]}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: List with complex tensors
    params7 = [torch.randn(2, 3, dtype=torch.complex64).numpy(), torch.randn(5, dtype=torch.complex64).numpy()]
    input_dict7 = {"parameters": params7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = parameters_to_vector_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('parameters_to_vector', generated_inputs)
