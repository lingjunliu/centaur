
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def Sigmoid_inputs():
    generated_inputs = []

    # Input 1: Scalar float
    input1 = np.array(0.5, dtype=np.float32)
    generated_inputs.append({"input": input1})

    # Input 2: 1D tensor with negative values
    input2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    generated_inputs.append({"input": input2})

    # Input 3: 2D tensor
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    generated_inputs.append({"input": input3})

    # Input 4: 3D tensor with mixed positive and negative values
    input4 = np.array([[[ -1, 2], [3, -4]], [[5, -6], [-7, 8]]], dtype=np.float32)
    generated_inputs.append({"input": input4})

    # Input 5: Larger tensor
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    generated_inputs.append({"input": input5})

    return generated_inputs

generated_inputs = Sigmoid_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Sigmoid', generated_inputs)
