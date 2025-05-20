
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def set_default_device_inputs():
    list_of_inputs = []

    input1 = {
        "device": "cpu"
    }
    list_of_inputs.append(copy.deepcopy(input1))

    if torch.cuda.is_available():
        input2 = {
            "device": "cuda"
        }
        list_of_inputs.append(copy.deepcopy(input2))

        input3 = {
            "device": "cuda:0"
        }
        list_of_inputs.append(copy.deepcopy(input3))
    
    if torch.backends.mps.is_available():
        input4 = {
            "device": "mps"
        }
        list_of_inputs.append(copy.deepcopy(input4))

    input5 = {
        "device": "meta"
    }
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs = set_default_device_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_default_device', generated_inputs)
