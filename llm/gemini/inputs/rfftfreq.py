
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def rfftfreq_inputs():
    list_of_inputs = []

    input1 = {
        "n": np.int32(5),
        "d": np.float64(1.0),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "n": np.int64(4),
        "d": np.float32(0.5),
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "n": np.int32(10),
        "d": np.float64(2.0),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {
        "n": np.int64(7),
        "d": np.float32(1.0),
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {
        "n": np.int32(16),
        "d": np.float64(0.1),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input5))

    input6 = {
        "n": np.int64(32),
        "d": np.float32(0.25),
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input6))

    input7 = {
        "n": np.int32(64),
        "d": np.float64(0.05),
        "device": None
    }
    list_of_inputs.append(copy.deepcopy(input7))

    return list_of_inputs

generated_inputs = rfftfreq_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rfftfreq', generated_inputs)
