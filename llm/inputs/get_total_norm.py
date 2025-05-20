
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def get_total_norm_inputs():
    list_of_inputs = []

    # Input 1: List of float tensors, norm_type=2.0
    tensor_list_1 = [torch.randn(3, 4).numpy(), torch.randn(5, 2).numpy()]
    input_dict_1 = {"parameters": tensor_list_1, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: List of float tensors, norm_type=1.0
    tensor_list_2 = [torch.randn(2, 2).numpy(), torch.randn(3,).numpy()]
    input_dict_2 = {"parameters": tensor_list_2, "norm_type": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: List of tensors with different shapes, norm_type=float('inf')
    tensor_list_3 = [torch.randn(1, 2, 3).numpy(), torch.randn(4).numpy(), torch.randn(2, 1).numpy()]
    input_dict_3 = {"parameters": tensor_list_3, "norm_type": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: List containing single tensor, norm_type=0.5
    tensor_list_4 = [torch.randn(10).numpy()]
    input_dict_4 = {"parameters": tensor_list_4, "norm_type": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: List of tensors with negative values, norm_type=3.0
    tensor_list_5 = [torch.randn(2, 3).numpy() * -1, torch.randn(4).numpy() * -1]
    input_dict_5 = {"parameters": tensor_list_5, "norm_type": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty list of tensors, norm_type=2.0
    tensor_list_6 = []
    input_dict_6 = {"parameters": tensor_list_6, "norm_type": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict_6))


    return list_of_inputs

generated_inputs = get_total_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('get_total_norm', generated_inputs)
