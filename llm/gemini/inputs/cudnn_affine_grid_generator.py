
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cudnn_affine_grid_generator_inputs():
    list_of_inputs = []

    theta1 = torch.randn(1, 2, 3).numpy()
    N1 = 1
    C1 = 1
    H1 = 10
    W1 = 20
    input_dict1 = {
        "theta": theta1,
        "N": N1,
        "C": C1,
        "H": H1,
        "W": W1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))


    return list_of_inputs

generated_inputs = cudnn_affine_grid_generator_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cudnn_affine_grid_generator', generated_inputs)
