
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def isreal_inputs():
    list_of_inputs = []

    # Input 1: Integer tensor
    input1 = torch.tensor([1, 2, 3, -4, 5]).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor
    input2 = torch.tensor([1.0, 2.5, -3.2, 4.7, 5.0]).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor with zero imaginary parts
    input3 = torch.tensor([1 + 0j, 2 + 0j, 3 + 0j, -4 + 0j, 5 + 0j]).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Complex tensor with non-zero imaginary parts
    input4 = torch.tensor([1 + 1j, 2 - 2j, 3 + 0.5j, -4 - 1j, 5 + 2j]).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Multidimensional tensor (2D) with mixed types
    input5 = torch.tensor([[1, 2.0, 3 + 0j], [4, -5.0, 6 + 1j]]).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = torch.tensor([]).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Multidimensional tensor (3D) with mixed types
    input7 = torch.tensor([[[1, 2.0, 3 + 0j], [4, -5.0, 6 + 1j]],[[7, 8.0, 9 + 0j], [10, -11.0, 12 + 1j]]]).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = isreal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('isreal', generated_inputs)
