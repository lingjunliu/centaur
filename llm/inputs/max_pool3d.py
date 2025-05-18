
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def max_pool3d_inputs():
    list_of_inputs = []
    input = torch.randn(2, 3, 10, 10, 10).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 4, 12, 12, 12).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = True
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 8, 8, 8).numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 2
    ceil_mode = False
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 1, 15, 15, 15).numpy()
    kernel_size = 4
    stride = 3
    padding = 1
    dilation = 1
    ceil_mode = True
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 8, 20, 20, 20).numpy()
    kernel_size = 5
    stride = 2
    padding = 2
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = max_pool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max_pool3d', list_of_inputs)
