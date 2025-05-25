
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def index_add_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensor
    input1 = np.zeros((5, 3), dtype=np.float32)
    index1 = np.array([0, 2, 4], dtype=np.int64)
    source1 = np.random.randn(3, 3).astype(np.float32)
    input_dict1 = {"input": input1, "dim": 0, "index": index1, "source": source1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = index_add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('index_add', generated_inputs)
