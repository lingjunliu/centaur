
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logical_or_inputs():
    list_of_inputs = []

    # Case 1: Basic boolean tensors
    input1 = (torch.randn(2, 3) > 0).numpy()
    input2 = (torch.randn(2, 3) > 0).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different shapes, but broadcastable
    input1 = (torch.randn(2, 1) > 0).numpy()
    input2 = (torch.randn(1, 3) > 0).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Scalars
    input1 = (torch.randn(1) > 0).numpy()
    input2 = (torch.randn(1) > 0).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: 3D tensors
    input1 = (torch.randn(2, 3, 4) > 0).numpy()
    input2 = (torch.randn(2, 3, 4) > 0).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Different dtypes (bool)
    input1 = (torch.randn(2, 3) > 0).numpy()
    input2 = (torch.randint(0, 2, (2, 3), dtype=torch.int32) > 0).numpy()
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Broadcasting with scalar
    input1 = (torch.randn(2, 3) > 0).numpy()
    input2 = np.array(True)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Broadcasting with scalar (False)
    input1 = (torch.randn(2, 3) > 0).numpy()
    input2 = np.array(False)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = logical_or_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logical_or', generated_inputs)
