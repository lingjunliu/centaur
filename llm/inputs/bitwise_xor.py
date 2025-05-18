
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bitwise_xor_inputs():
    list_of_inputs = []

    input1 = torch.randint(0, 2, (2, 3), dtype=torch.int8).numpy()
    input2 = torch.randint(0, 2, (2, 3), dtype=torch.int8).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(0, 256, (4, 4), dtype=torch.int16).numpy()
    input2 = torch.randint(0, 256, (4, 4), dtype=torch.int16).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(0, 1024, (1, 5, 5), dtype=torch.int32).numpy()
    input2 = torch.randint(0, 1024, (1, 5, 5), dtype=torch.int32).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(0, 65536, (2, 2, 2, 2), dtype=torch.int64).numpy()
    input2 = torch.randint(0, 65536, (2, 2, 2, 2), dtype=torch.int64).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(0, 2, (3, 3), dtype=torch.uint8).numpy()
    input2 = torch.randint(0, 2, (3, 3), dtype=torch.uint8).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = bitwise_xor_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bitwise_xor', list_of_inputs)
