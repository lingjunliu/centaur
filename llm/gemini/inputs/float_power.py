
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def float_power_inputs():
    list_of_inputs = []

    # Case 1: Basic float input and exponent
    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer input and float exponent
    input_tensor = torch.randint(1, 5, (3, 4)).numpy()
    exponent_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Float input and integer exponent
    input_tensor = torch.randn(4, 2).numpy()
    exponent_tensor = torch.randint(1, 4, (4, 2)).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Scalar input and exponent
    input_tensor = torch.randn(1).item()
    exponent_tensor = torch.randn(1).item()
    input_dict = {"input": np.array(input_tensor), "exponent":  np.array(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Negative input and float exponent (check behavior)
    input_tensor = torch.randn(2, 2) * -1.0
    input_tensor = input_tensor.numpy()
    exponent_tensor = torch.randn(2, 2).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Different shaped inputs (broadcastable)
    input_tensor = torch.randn(2, 3).numpy()
    exponent_tensor = torch.randn(1, 3).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Multi-dimensional input
    input_tensor = torch.randn(2, 3, 4).numpy()
    exponent_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "exponent": exponent_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = float_power_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('float_power', generated_inputs)
