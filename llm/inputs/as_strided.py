
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def as_strided_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D float tensor
    input_tensor = torch.randn(5, 7).numpy()
    size = (3, 4)
    stride = (7, 1)
    storage_offset = 0
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 3D int tensor with offset
    input_tensor = torch.randint(0, 10, (4, 5, 6)).numpy()
    size = (2, 3, 4)
    stride = (30, 6, 1)
    storage_offset = 7
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 1D tensor
    input_tensor = torch.arange(10).float().numpy()
    size = (5,)
    stride = (2,)
    storage_offset = 1
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Larger 2D tensor with different strides
    input_tensor = torch.randn(10, 12).numpy()
    size = (5, 5)
    stride = (12, 2)
    storage_offset = 3
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Bool Tensor
    input_tensor = torch.randint(0, 2, (3, 3), dtype=torch.bool).numpy()
    size = (2, 2)
    stride = (3, 1)
    storage_offset = 0
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Complex Tensor
    input_tensor = torch.randn(3, 3, dtype=torch.complex64).numpy()
    size = (2, 2)
    stride = (3, 1)
    storage_offset = 0
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: 4D tensor
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    size = (1, 2, 2, 3)
    stride = (60, 20, 5, 1)
    storage_offset = 2
    input_dict = {"input": input_tensor, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = as_strided_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('as_strided', generated_inputs)
