
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def moveaxis_inputs():
    list_of_inputs = []

    # Case 1: Simple 3D tensor, moving axis 1 to 0
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 4D tensor, moving axis 0 to 2
    t = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": t, "source": 0, "destination": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D tensor, moving axis 2 to -1 (same as 2)
    t = torch.randn(3, 2, 1).numpy()
    input_dict = {"input": t, "source": 2, "destination": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: 5D tensor, move axis -1 to axis 0
    t = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict = {"input": t, "source": -1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 2D tensor, move axis 0 to axis 1
    t = torch.randn(2, 3).numpy()
    input_dict = {"input": t, "source": 0, "destination": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Float64 tensor, moving axis 1 to 0
    t = torch.randn(3, 2, 1, dtype=torch.float64).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Int tensor
    t = torch.randint(0, 10, (3, 2, 1)).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: Complex tensor
    t = torch.randn(3, 2, 1, dtype=torch.complex64).numpy()
    input_dict = {"input": t, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = moveaxis_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('moveaxis', generated_inputs)
