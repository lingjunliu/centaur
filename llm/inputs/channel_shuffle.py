
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def channel_shuffle_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 4D tensor and groups = 2
    input1 = torch.randn(1, 4, 10, 10).numpy()
    groups1 = 2
    input_dict1 = {"input": input1.astype(np.float32), "groups": groups1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 4D tensor with groups equal to number of channels
    input2 = torch.randn(1, 8, 8, 8).numpy()
    groups2 = 8
    input_dict2 = {"input": input2.astype(np.float32), "groups": groups2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor with different batch size and different channel size
    input3 = torch.randn(2, 6, 5, 5).numpy()
    groups3 = 3
    input_dict3 = {"input": input3.astype(np.float32), "groups": groups3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D tensor with a larger group size
    input4 = torch.randn(1, 16, 4, 4).numpy()
    groups4 = 4
    input_dict4 = {"input": input4.astype(np.float32), "groups": groups4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs = channel_shuffle_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('channel_shuffle', generated_inputs)
