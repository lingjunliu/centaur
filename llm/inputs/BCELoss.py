
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def bceloss_inputs():
    generated_inputs = []

    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean'
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "weight": torch.randn(5).numpy(),
        "size_average": True,
        "reduce": False,
        "reduction": None
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = bceloss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('BCELoss', generated_inputs)
