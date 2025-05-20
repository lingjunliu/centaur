
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def typename_inputs():
    list_of_inputs = []

    # Input 1: Float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int tensor
    input_tensor = torch.randint(0, 10, (2, 2)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Bool tensor
    input_tensor = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Complex tensor
    input_tensor = torch.complex(torch.randn(2, 3), torch.randn(2, 3)).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Long tensor
    input_tensor = torch.randint(0, 100, (5,)).long().numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Double tensor
    input_tensor = torch.randn(2, 3, dtype=torch.float64).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero dimensional tensor (scalar)
    input_tensor = torch.tensor(5.0).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = typename_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('typename', generated_inputs)
