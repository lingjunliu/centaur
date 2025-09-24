
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def clone_inputs():
    list_of_inputs = []

    # Example 1: 1D Float Tensor
    input_1 = torch.randn(5).numpy()
    input_dict_1 = {"input": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Example 2: 2D Int Tensor
    input_2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict_2 = {"input": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Example 3: 3D Complex Tensor
    input_3 = (torch.randn(2, 3, 2) + 1j * torch.randn(2, 3, 2)).numpy()
    input_dict_3 = {"input": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Example 4: 4D Float Tensor with negative values
    input_4 = torch.randn(1, 3, 2, 2) * -1.0
    input_4 = input_4.numpy()
    input_dict_4 = {"input": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Example 5: Scalar Tensor
    input_5 = torch.tensor(5.0).numpy()
    input_dict_5 = {"input": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Example 6: Empty Tensor
    input_6 = torch.empty(0).numpy()
    input_dict_6 = {"input": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Example 7: Bool Tensor
    input_7 = torch.tensor([True, False, True]).numpy()
    input_dict_7 = {"input": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs = clone_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('clone', generated_inputs)
