
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_unpool3d_inputs():
    list_of_inputs = []

    # Example 1
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    input_tensor = torch.randn(5, 16, 16, 16, 16)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": input_tensor.shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = max_unpool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxUnpool3d', generated_inputs)
