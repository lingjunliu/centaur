
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def pow_inputs():
    list_of_inputs = []

    # Case 1: Float input and float exponent
    input_tensor = np.random.rand(2, 3).astype(np.float32)
    exponent_tensor = np.array(2.0).astype(np.float32)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Int input and int exponent
    input_tensor = np.random.randint(1, 5, size=(3, 4), dtype=np.int32)
    exponent_tensor = np.array(3).astype(np.int32)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Float input and float tensor exponent
    input_tensor = np.random.rand(4, 2).astype(np.float64)
    exponent_tensor = np.random.rand(4, 2).astype(np.float64)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: Int input and float exponent
    input_tensor = np.random.randint(-5, 5, size=(2, 2), dtype=np.int64)
    exponent_tensor = np.array(0.5).astype(np.float64)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.tensor(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative input and float exponent
    input_tensor = np.random.randn(3, 3).astype(np.float32) * -1
    exponent_tensor = np.array(2.0).astype(np.float32)
    input_dict = {"input": torch.from_numpy(input_tensor), "exponent": torch.from_numpy(exponent_tensor)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = pow_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pow', generated_inputs)
