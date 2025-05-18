
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def is_nonzero_inputs():
    list_of_inputs = []

    input1 = torch.randn(1).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.zeros(1).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.0]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1.0]).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randint(-5,5,(1,)).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = is_nonzero_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('is_nonzero', list_of_inputs)
