
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def slogdet_inputs():
    list_of_inputs = []

    A = torch.randn(3, 3).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(2, 2).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(5, 5).numpy()
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = slogdet_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('slogdet', generated_inputs)
