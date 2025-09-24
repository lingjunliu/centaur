
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def intstorage_inputs():
    list_of_inputs = []

    input1 = {
        "storage": torch.IntStorage(np.array([1, 2, 3, 4, 5], dtype=np.int64).tolist())
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "storage": torch.IntStorage(np.array([], dtype=np.int64).tolist())
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "storage": torch.IntStorage(np.array([-1, 0, 1], dtype=np.int64).tolist())
    }
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {
        "storage": torch.IntStorage(np.array([2**31 - 1, -2**31], dtype=np.int64).tolist())
    }
    list_of_inputs.append(copy.deepcopy(input4))
    
    input5 = {
        "storage": torch.IntStorage(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64).tolist())
    }
    list_of_inputs.append(copy.deepcopy(input5))
    
    return list_of_inputs

generated_inputs = intstorage_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('IntStorage', generated_inputs)
