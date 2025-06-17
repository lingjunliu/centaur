
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def as_strided_copy_inputs():
    list_of_inputs = []

    # Case 1: Simple 2D float tensor
    source = torch.randn(5, 5).numpy()
    size = (3, 3)
    stride = (1, 1)
    storage_offset = 0
    out = torch.empty(size, dtype=torch.float32).numpy()

    input_dict = {
        "source": source,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D int tensor with non-zero storage offset
    source = torch.randint(0, 10, (4, 4, 4)).numpy()
    size = (2, 2, 2)
    stride = (1, 1, 1)
    storage_offset = 5
    out = torch.empty(size, dtype=torch.int64).numpy()

    input_dict = {
        "source": source,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D tensor with larger strides
    source = torch.arange(10).float().numpy()
    size = (3,)
    stride = (2,)
    storage_offset = 1
    out = torch.empty(size, dtype=torch.float32).numpy()

    input_dict = {
        "source": source,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D tensor with different strides
    source = torch.randn(6, 6).numpy()
    size = (3, 3)
    stride = (2, 1)
    storage_offset = 0
    out = torch.empty(size, dtype=torch.float32).numpy()

    input_dict = {
        "source": source,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Complex 2D tensor
    source = torch.randn(4, 4, dtype=torch.complex64).numpy()
    size = (2, 2)
    stride = (1, 1)
    storage_offset = 0
    out = torch.empty(size, dtype=torch.complex64).numpy()

    input_dict = {
        "source": source,
        "size": size,
        "stride": stride,
        "storage_offset": storage_offset,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.as_strided_copy"] = as_strided_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.as_strided_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_strided_copy'.")

check_valid('torch.as_strided_copy', generated_inputs['torch.as_strided_copy'], lib="torch")
