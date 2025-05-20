
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def as_strided_copy_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D float tensor
    input1 = torch.randn(5, 5).numpy()
    size1 = (3, 3)
    stride1 = (1, 1)
    input_dict1 = {"input": input1, "size": size1, "stride": stride1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 3D int tensor
    input2 = torch.randint(0, 10, (4, 4, 4)).numpy()
    size2 = (2, 2, 2)
    stride2 = (1, 1, 2)
    input_dict2 = {"input": input2, "size": size2, "stride": stride2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 1D tensor
    input3 = torch.arange(10).float().numpy()
    size3 = (5,)
    stride3 = (2,)
    input_dict3 = {"input": input3, "size": size3, "stride": stride3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Case 4: Different size and stride combination for 2D tensor
    input4 = torch.randn(7, 7).numpy()
    size4 = (4, 4)
    stride4 = (2, 1)
    input_dict4 = {"input": input4, "size": size4, "stride": stride4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Larger tensor and size with complex strides
    input5 = torch.randn(10, 10, 10).numpy()
    size5 = (5, 5, 5)
    stride5 = (3, 2, 1)
    input_dict5 = {"input": input5, "size": size5, "stride": stride5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = as_strided_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('as_strided_copy', generated_inputs)
