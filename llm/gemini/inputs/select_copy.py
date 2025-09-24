
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def select_copy_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D float tensor
    input_tensor = torch.randn(3, 4).numpy()
    dim = 0
    index = 1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D int tensor, negative index
    input_tensor = torch.randint(-5, 5, (2, 3, 5)).numpy()
    dim = 1
    index = -1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 4D complex tensor
    input_tensor = torch.randn(2, 2, 2, 2, dtype=torch.complex64).numpy()
    dim = 2
    index = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 4: 1D tensor
    input_tensor = torch.arange(5).float().numpy()
    dim = 0
    index = 2
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Large 3D tensor
    input_tensor = torch.randn(10, 20, 30).numpy()
    dim = 0
    index = 5
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Boolean tensor
    input_tensor = torch.randint(0, 2, (2, 3), dtype=torch.bool).numpy()
    dim = 1
    index = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Double tensor
    input_tensor = torch.randn(3, 3, dtype=torch.float64).numpy()
    dim = 0
    index = 1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = select_copy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('select_copy', generated_inputs)
