
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def get_device_inputs():
    list_of_inputs = []

    # Input 1: Float tensor on CPU
    input_tensor_float = torch.randn(2, 3, 4, 5)
    input_dict_1 = {"input": input_tensor_float.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Int tensor on CPU
    input_tensor_int = torch.randint(0, 10, (3, 2, 5))
    input_dict_2 = {"input": input_tensor_int.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Long tensor on CPU
    input_tensor_long = torch.randint(0, 100, (2, 2, 2), dtype=torch.long)
    input_dict_3 = {"input": input_tensor_long.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Bool tensor on CPU
    input_tensor_bool = torch.randint(0, 2, (4, 4)).bool()
    input_dict_4 = {"input": input_tensor_bool.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Double tensor on CPU
    input_tensor_double = torch.randn(5, 5, dtype=torch.float64)
    input_dict_5 = {"input": input_tensor_double.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: One-dimensional tensor
    input_tensor_1d = torch.randn(10)
    input_dict_6 = {"input": input_tensor_1d.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Zero-dimensional tensor
    input_tensor_0d = torch.tensor(5.0)
    input_dict_7 = {"input": input_tensor_0d.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensor with large values
    input_tensor_large = torch.randn(2, 2) * 1e5
    input_dict_8 = {"input": input_tensor_large.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    return list_of_inputs

generated_inputs = get_device_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('get_device', generated_inputs)
