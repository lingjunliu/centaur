
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def hermite_polynomial_he_inputs():
    generated_inputs = []

    x1 = np.array([1.0, 2.0, 3.0])
    n1 = 2
    input_dict1 = {"x": x1, "n": n1}
    generated_inputs.append(copy.deepcopy(input_dict1))
    
    return generated_inputs

generated_inputs = hermite_polynomial_he_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hermite_polynomial_he', generated_inputs)
