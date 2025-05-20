
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def fftfreq_inputs():
    list_of_inputs = []

    input_dict = {
        "n": 5,
        "d": 1.0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 4,
        "d": 0.5,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 10,
        "d": 2.0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 7,
        "d": 0.1,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 16,
        "d": 1.5,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 3,
        "d": 1.0,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "n": 8,
        "d": 0.25,
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = fftfreq_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fftfreq', generated_inputs)
