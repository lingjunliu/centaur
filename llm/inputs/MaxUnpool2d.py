
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 1, 2, 2).numpy()
    indices = torch.randint(0, 4, (1, 1, 2, 2)).long().numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    output_size = None
    
    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 3, 3).numpy()
    indices = torch.randint(0, 9, (1, 3, 3, 3)).long().numpy()
    kernel_size = 3
    stride = 1
    padding = 0
    output_size = None
    
    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 4, 4).numpy()
    indices = torch.randint(0, 16, (1, 2, 4, 4)).long().numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    output_size = (9,9)
    
    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 1, 5, 5).numpy()
    indices = torch.randint(0, 25, (2, 1, 5, 5)).long().numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    output_size = None
    
    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 8, 10, 10).numpy()
    indices = torch.randint(0, 100, (2, 8, 10, 10)).long().numpy()
    kernel_size = 4
    stride = 2
    padding = 1
    output_size = (21, 21)
    
    input_dict = {
        "input": input,
        "indices": indices,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = max_unpool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxUnpool2d', list_of_inputs)
