
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rand_inputs():
    generated_inputs = []

    generated_inputs.append({"size": np.array([1]).item()})
    generated_inputs.append({"size": np.array([2, 3]).tolist()})
    generated_inputs.append({"size": np.array([4, 5, 6]).tolist()})
    generated_inputs.append({"size": np.array([2, 3, 4, 5]).tolist()})
    generated_inputs.append({"size": np.array([1, 1, 1, 1, 1]).tolist()})
    generated_inputs.append({"size": np.array([10]).item()})
    generated_inputs.append({"size": np.array([2, 7]).tolist()})
    generated_inputs.append({"size": np.array([3, 1, 5]).tolist()})
    generated_inputs.append({"size": np.array([6, 2, 8, 3]).tolist()})
    generated_inputs.append({"size": np.array([1, 2, 3, 4, 5, 6]).tolist()})
    generated_inputs.append({"size": np.array([2, 1, 4]).tolist()})

    return generated_inputs

generated_inputs = rand_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rand', generated_inputs)
