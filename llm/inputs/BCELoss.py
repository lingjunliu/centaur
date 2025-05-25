
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def bceloss_inputs():
    list_of_inputs = []

    # Case 1: No weight, default reduction
    input_dict = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "input": np.random.rand(3,2) ,
        "target": np.random.rand(3,2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: weight, reduction='sum'
    input_dict = {
        "weight": torch.randn(5).numpy() if torch.cuda.is_available() else torch.randn(5).cpu().numpy(),
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "input": np.random.rand(5),
        "target": np.random.rand(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = bceloss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('BCELoss', generated_inputs)
