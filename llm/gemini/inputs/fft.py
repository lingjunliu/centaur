
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def fft_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(4).numpy()
    input_dict1 = {
        "input": input1,
        "n": None,
        "dim": -1,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor with specified n and dim
    input2 = torch.randint(0, 10, (8,)).numpy()
    input_dict2 = {
        "input": input2,
        "n": 4,
        "dim": 0,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor with norm
    input3 = torch.randn(5, dtype=torch.complex64).numpy()
    input_dict3 = {
        "input": input3,
        "n": 10,
        "dim": -1,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Multidimensional tensor
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {
        "input": input4,
        "n": None,
        "dim": 1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensor with negative values and specified length
    input5 = torch.randint(-5, 5, (7,)).float().numpy()
    input_dict5 = {
        "input": input5,
        "n": 16,
        "dim": 0,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different dim
    input6 = torch.randn(2, 3, 4).numpy()
    input_dict6 = {
        "input": input6,
        "n": 2,
        "dim": 2,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger tensor
    input7 = torch.randn(10).numpy()
    input_dict7 = {
        "input": input7,
        "n": 5,
        "dim": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Complex tensor with different dim
    input8 = torch.randn(2, 4, dtype=torch.complex64).numpy()
    input_dict8 = {
        "input": input8,
        "n": None,
        "dim": 1,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = fft_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fft', generated_inputs)
