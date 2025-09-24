
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def modified_bessel_k0_inputs():
    generated_inputs = []

    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    generated_inputs.append({"x": x1})

    return generated_inputs

generated_inputs = modified_bessel_k0_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('modified_bessel_k0', generated_inputs)
