
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def as_strided_inputs():
    list_of_inputs = []

    # Test case 1: Basic 1D tensor
    input = np.arange(10, dtype=np.float32)
    size = (5,)
    stride = (2,)
    storage_offset = 0
    input_dict = {"input": input, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor
    input = np.arange(20, dtype=np.int64).reshape(4, 5)
    size = (2, 3)
    stride = (5, 1)
    storage_offset = 0
    input_dict = {"input": input, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 3D tensor with offset
    input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    size = (1, 2, 2)
    stride = (12, 4, 1)
    storage_offset = 3
    input_dict = {"input": input, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different data type (complex)
    input = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j], dtype=np.complex128)
    size = (3,)
    stride = (1,)
    storage_offset = 1
    input_dict = {"input": input, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Larger tensor with more complex size and stride
    input = np.arange(100, dtype=np.float32).reshape(10, 10)
    size = (3, 3)
    stride = (10, 1)
    storage_offset = 0
    input_dict = {"input": input, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 7: bool tensor
    input = np.array([True, False, True, True, False], dtype=bool)
    size = (3,)
    stride = (1,)
    storage_offset = 0
    input_dict = {"input": input, "size": size, "stride": stride, "storage_offset": storage_offset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_strided"] = as_strided_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.as_strided' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_strided'.")

check_valid('torch.as_strided', generated_inputs['torch.as_strided'], lib="torch")
