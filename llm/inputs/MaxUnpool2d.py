
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 1, 2, 2).numpy()
    indices1 = torch.tensor([[0, 1], [2, 3]]).numpy()
    kernel_size1 = 2
    stride1 = 2
    padding1 = 0
    output_size1 = None
    input_dict1 = {
        "input": input1,
        "indices": indices1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "output_size": output_size1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

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
