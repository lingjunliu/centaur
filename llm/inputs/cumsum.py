
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cumsum_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor, dim=0
    input1 = torch.randint(1, 20, (10,)).numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D tensor, dim=0
    input2 = torch.randn(5, 3).numpy()
    dim2 = 0
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 2D tensor, dim=1
    input3 = torch.randn(5, 3).numpy()
    dim3 = 1
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 3D tensor, dim=2
    input4 = torch.randn(2, 4, 5).numpy()
    dim4 = 2
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Tensor with negative values, dim=0
    input5 = torch.randint(-10, 10, (7,)).float().numpy()
    dim5 = 0
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: Tensor with integer dtype
    input6 = torch.randint(-5, 5, (3,4)).int().numpy()
    dim6 = 1
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Example 7: 4D tensor, dim = 1
    input7 = torch.randn(2, 3, 4, 5).numpy()
    dim7 = 1
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = cumsum_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cumsum', generated_inputs)
