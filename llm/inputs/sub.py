
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def sub_inputs():
    list_of_inputs = []

    # Input 1: Basic subtraction with float tensors
    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    alpha1 = 1.0
    input_dict1 = {"input": input1, "other": other1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Subtraction with integer tensors
    input2 = torch.randint(0, 10, (2, 2), dtype=torch.int32).numpy()
    other2 = torch.randint(0, 5, (2, 2), dtype=torch.int32).numpy()
    alpha2 = 1
    input_dict2 = {"input": input2, "other": other2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Subtraction with different shaped tensors (broadcasting)
    input3 = torch.randn(5, 1).numpy()
    other3 = torch.randn(1, 5).numpy()
    alpha3 = 1.0
    input_dict3 = {"input": input3, "other": other3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Subtraction with scalar other and alpha
    input4 = torch.randn(2, 3, 4).numpy()
    other4 = np.float64(2.0)
    alpha4 = 0.5
    input_dict4 = {"input": input4, "other": other4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Subtraction with complex tensors
    input5 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    other5 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    alpha5 = 1.0
    input_dict5 = {"input": input5, "other": other5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = sub_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sub', generated_inputs)
