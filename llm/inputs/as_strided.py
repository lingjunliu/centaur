
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def as_strided_inputs():
    list_of_inputs = []

    # Example 1
    input_tensor = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6).numpy()
    size = (1, 3)
    stride = (0, 2)
    storage_offset = 0

    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    input_tensor = torch.arange(1, 17, dtype=torch.float32).reshape(1, 16).numpy()
    size = (2, 3)
    stride = (6, 2)
    storage_offset = 0

    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    input_tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3).numpy()
    size = (2, 2)
    stride = (1, 1)
    storage_offset = 0

    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4
    input_tensor = torch.arange(24).reshape(2, 3, 4).numpy()
    size = (2, 2, 2)
    stride = (12, 4, 1)
    storage_offset = 0

    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5
    input_tensor = torch.arange(10).numpy()
    size = (5,)
    stride = (2,)
    storage_offset = 0

    input_dict = {
        "input": input_tensor,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = as_strided_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('as_strided', list_of_inputs)
