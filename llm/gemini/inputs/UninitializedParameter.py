
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch
import copy
import numpy as np

def uninitialized_parameter_inputs():
    list_of_inputs = []

    input_dict = {
        "dtype": torch.float32,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.UninitializedParameter"] = uninitialized_parameter_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('UninitializedParameter', generated_inputs['torch.nn.UninitializedParameter'], lib="torch")
