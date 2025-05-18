
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    input_np = torch.randn(1, 1, 2, 2).numpy()
    indices_np = torch.randint(0, 4, (1, 1, 2, 2)).long().numpy()
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    output_size = (4, 4)
    
    input_dict = {
        "input": input_np,
        "indices": indices_np,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(1, 3, 3, 3).numpy()
    indices_np = torch.randint(0, 9, (1, 3, 3, 3)).long().numpy()
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    output_size = (5, 5)
    
    input_dict = {
        "input": input_np,
        "indices": indices_np,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(2, 2, 4, 4).numpy()
    indices_np = torch.randint(0, 16, (2, 2, 4, 4)).long().numpy()
    kernel_size = (4, 4)
    stride = (2, 2)
    padding = (1, 1)
    output_size = (7, 7)
    
    input_dict = {
        "input": input_np,
        "indices": indices_np,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(1, 1, 5, 5).numpy()
    indices_np = torch.randint(0, 25, (1, 1, 5, 5)).long().numpy()
    kernel_size = (5, 5)
    stride = (3, 3)
    padding = (1, 1)
    output_size = (11, 11)
    
    input_dict = {
        "input": input_np,
        "indices": indices_np,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_size": output_size
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_np = torch.randn(2, 4, 2, 2).numpy()
    indices_np = torch.randint(0, 4, (2, 4, 2, 2)).long().numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    output_size = (3, 3)
    
    input_dict = {
        "input": input_np,
        "indices": indices_np,
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

check_valid('max_unpool2d', list_of_inputs)
