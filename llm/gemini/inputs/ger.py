
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ger_inputs():
    generated_inputs = []

    # Test case 1: Basic float tensors
    vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec2 = np.array([4.0, 5.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})

    # Test case 2: Integer tensors
    vec1 = np.array([1, 2, 3], dtype=np.int32)
    vec2 = np.array([4, 5], dtype=np.int32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})

    # Test case 3: Negative values
    vec1 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    vec2 = np.array([4.0, -5.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})

    # Test case 4: Different sizes
    vec1 = np.array([1.0, 2.0], dtype=np.float32)
    vec2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})
    
    # Test case 5: Single element tensors
    vec1 = np.array([1.0], dtype=np.float32)
    vec2 = np.array([4.0], dtype=np.float32)
    generated_inputs.append({"vec1": vec1, "vec2": vec2})

    return generated_inputs

def check_valid(api_name, list_of_inputs):
    for input_dict in list_of_inputs:
        api_driver(input_dict, cpu=True)

def api_driver(input, cpu=True):
    vec1 = torch.tensor(input["vec1"])
    vec2 = torch.tensor(input["vec2"])
    
    if cpu:
        vec1 = vec1.cpu()
        vec2 = vec2.cpu()
    else:
        vec1 = vec1.cuda()
        vec2 = vec2.cuda()

    torch.ger(vec1, vec2)

generated_inputs = ger_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ger', generated_inputs)
