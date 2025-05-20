
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor with alpha=1.0
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "alpha": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor with negative values and alpha=0.5
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    input_dict = {"input": input_tensor, "alpha": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float tensor with alpha=2.0
    input_tensor = torch.randn(5,).numpy()
    input_dict = {"input": input_tensor, "alpha": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: High-dimensional tensor with alpha=0.25
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": input_tensor, "alpha": 0.25}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar tensor (0-dimensional) with alpha=1.5
    input_tensor = torch.randn(1).item()
    input_tensor = np.array(input_tensor)
    input_dict = {"input": input_tensor, "alpha": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = celu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('celu_', generated_inputs)
