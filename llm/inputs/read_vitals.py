
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def read_vitals_inputs():
    list_of_inputs = []

    input1 = {
        "path": "/tmp/test_file_1.txt",
        "devices": ["cpu"],
        "timeout": 10.0,
        "retry_count": 3
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "path": "/tmp/test_file_2.log",
        "devices": ["cuda:0", "cpu"],
        "timeout": 5.5,
        "retry_count": 1
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "path": "/home/user/data.dat",
        "devices": [],
        "timeout": 0.1,
        "retry_count": 5
    }
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {
        "path": "C:\\data\\important.vtl",
        "devices": ["cpu"],
        "timeout": 100.0,
        "retry_count": 0
    }
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {
        "path": "./relative/path/data.txt",
        "devices": ["cuda:1"],
        "timeout": 2.718,
        "retry_count": 2
    }
    list_of_inputs.append(copy.deepcopy(input5))

    input6 = {
        "path": "https://example.com/data.vitals",
        "devices": ["cpu", "cuda:0", "cuda:1"],
        "timeout": 1.618,
        "retry_count": 4
    }
    list_of_inputs.append(copy.deepcopy(input6))

    return list_of_inputs

generated_inputs = read_vitals_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('read_vitals', generated_inputs)
