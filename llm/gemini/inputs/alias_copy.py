
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def alias_copy_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor, 2D
    input2 = torch.randint(-10, 10, (3, 4)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor, 3D
    input3 = torch.randn(2, 3, 2, dtype=torch.complex64).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Bool tensor, 2D
    input4 = torch.randint(0, 2, (4, 5)).bool().numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Long tensor, 4D
    input5 = torch.randint(-5, 15, (1, 2, 3, 4), dtype=torch.int64).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Double tensor, 2D
    input6 = torch.randn(2, 2, dtype=torch.float64).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Byte tensor, 3D
    input7 = torch.randint(0, 256, (3, 2, 3), dtype=torch.uint8).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = alias_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('alias_copy', generated_inputs)
