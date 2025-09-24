
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def tanh_inputs():
    generated_inputs = []

    # Input 1: 1D tensor of floats
    input1 = np.random.randn(5).astype(np.float32)
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor of floats with negative values
    input2 = np.random.randn(3, 4).astype(np.float64)
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor of integers
    input3 = np.random.randint(-5, 5, size=(2, 3, 2)).astype(np.int32)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Scalar float
    input4 = np.array(3.14).astype(np.float32)
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor of complex numbers
    input5 = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    input_dict5 = {"input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = np.array([]).astype(np.float32)
    input_dict6 = {"input": input6}
    generated_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 1D tensor of floats with large values
    input7 = np.array([-1000.0, 0.0, 1000.0]).astype(np.float32)
    input_dict7 = {"input": input7}
    generated_inputs.append(copy.deepcopy(input_dict7))

    return generated_inputs

generated_inputs = tanh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('tanh', generated_inputs)
