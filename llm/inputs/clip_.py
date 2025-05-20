
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def clip_inputs():
    list_of_inputs = []

    input_float = np.random.randn(3, 4).astype(np.float32)
    min_val = -0.5
    max_val = 1.5
    input_dict = {"input": input_float, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_int = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    min_val = -2
    max_val = 3
    input_dict = {"input": input_int.astype(np.float32), "min": float(min_val), "max": float(max_val)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_neg = np.random.randn(1, 5) * -1.0
    input_neg = input_neg.astype(np.float64)
    min_val = -2.0
    max_val = -0.5
    input_dict = {"input": input_neg, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_3d = np.random.rand(2, 3, 4).astype(np.float32)
    min_val = 0.2
    max_val = 0.8
    input_dict = {"input": input_3d, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_bool = np.random.choice([True, False], size=(2, 3)).astype(np.float32)
    min_val = 0.0
    max_val = 1.0
    input_dict = {"input": input_bool, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = clip_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('clip_', generated_inputs)
