
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def Tanhshrink_inputs():
    generated_inputs = []

    input1 = np.random.randn(2).astype(np.float32)
    generated_inputs.append({"input": input1})

    input2 = np.random.randn(2, 3).astype(np.float64)
    generated_inputs.append({"input": input2})

    input3 = np.random.randn(2, 3, 4).astype(np.float32)
    generated_inputs.append({"input": input3})

    input4 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]).astype(np.float32)
    generated_inputs.append({"input": input4})
    
    input5 = np.random.randn(1, 1, 5, 5).astype(np.float64)
    generated_inputs.append({"input": input5})

    return generated_inputs

generated_inputs = Tanhshrink_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Tanhshrink', generated_inputs)
