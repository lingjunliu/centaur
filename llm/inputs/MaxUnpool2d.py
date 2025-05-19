
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with specified output size
    input1 = torch.randn(1, 1, 2, 2).numpy()
    indices1 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size1 = 2
    stride1 = 2
    padding1 = 0
    output_size1 = (5, 5)

    input_dict1 = {
        "input": input1,
        "indices": indices1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "output_size": output_size1
    }

    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different kernel size and stride
    input2 = torch.randn(1, 3, 3, 3).numpy()
    indices2 = torch.randint(0, 9, (1, 3, 3, 3)).numpy()
    kernel_size2 = 3
    stride2 = 1
    padding2 = 1
    output_size2 = None

    input_dict2 = {
        "input": input2,
        "indices": indices2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "output_size": output_size2
    }

    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Non-square kernel size
    input3 = torch.randn(1, 1, 3, 4).numpy()
    indices3 = torch.randint(0, 12, (1, 1, 3, 4)).numpy()
    kernel_size3 = (3, 2)
    stride3 = (2, 1)
    padding3 = (1, 0)
    output_size3 = None

    input_dict3 = {
        "input": input3,
        "indices": indices3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "output_size": output_size3
    }

    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger input with batch size > 1
    input4 = torch.randn(2, 4, 5, 5).numpy()
    indices4 = torch.randint(0, 25, (2, 4, 5, 5)).numpy()
    kernel_size4 = 2
    stride4 = 2
    padding4 = 0
    output_size4 = (10, 10)

    input_dict4 = {
        "input": input4,
        "indices": indices4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "output_size": output_size4
    }

    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Output size is None
    input5 = torch.randn(1, 2, 4, 4).numpy()
    indices5 = torch.randint(0, 16, (1, 2, 4, 4)).numpy()
    kernel_size5 = 4
    stride5 = 4
    padding5 = 0
    output_size5 = None

    input_dict5 = {
        "input": input5,
        "indices": indices5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "output_size": output_size5
    }

    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Single int for kernel_size, stride, padding
    input6 = torch.randn(1, 1, 2, 2).numpy()
    indices6 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size6 = 2
    stride6 = 1
    padding6 = 0
    output_size6 = None

    input_dict6 = {
        "input": input6,
        "indices": indices6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "output_size": output_size6
    }

    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Tuple for kernel_size, stride, padding
    input7 = torch.randn(1, 1, 2, 2).numpy()
    indices7 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    kernel_size7 = (2, 2)
    stride7 = (1, 1)
    padding7 = (0, 0)
    output_size7 = None

    input_dict7 = {
        "input": input7,
        "indices": indices7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "output_size": output_size7
    }

    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs = max_unpool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxUnpool2d', generated_inputs)
