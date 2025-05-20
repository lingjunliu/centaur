
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def rfft_inputs():
    generated_inputs = []

    # Test case 1: Basic test with a 1D tensor
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict1 = {"input": input1, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D tensor with specified dim
    input2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict2 = {"input": input2, "n": None, "dim": 0, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Specify n (signal length) - padding
    input3 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    input_dict3 = {"input": input3, "n": 5, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Specify n (signal length) - trimming
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict4 = {"input": input4, "n": 3, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: With normalization
    input5 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict5 = {"input": input5, "n": None, "dim": -1, "norm": "forward"}
    generated_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: With different normalization
    input6 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict6 = {"input": input6, "n": None, "dim": -1, "norm": "ortho"}
    generated_inputs.append(copy.deepcopy(input_dict6))
    
    # Test case 7: With negative values
    input7 = torch.tensor([-1.0, 2.0, -3.0, 4.0]).numpy()
    input_dict7 = {"input": input7, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict7))

    # Test case 8: 3D tensor
    input8 = torch.randn(2, 3, 4).numpy()
    input_dict8 = {"input": input8, "n": None, "dim": 1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict8))
    
    # Test case 9: float64 tensor
    input9 = torch.randn(4, dtype=torch.float64).numpy()
    input_dict9 = {"input": input9, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict9))
    
    # Test case 10: int64 tensor
    input10 = torch.randint(0, 10, (4,), dtype=torch.int64).numpy()
    input_dict10 = {"input": input10, "n": None, "dim": -1, "norm": None}
    generated_inputs.append(copy.deepcopy(input_dict10))

    return generated_inputs

generated_inputs = rfft_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rfft', generated_inputs)
