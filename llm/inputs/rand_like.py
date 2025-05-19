
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rand_like_inputs():
    generated_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    generated_inputs.append({"input": input1})

    input2 = np.random.randint(0, 10, size=(2, 2), dtype=np.int64)
    generated_inputs.append({"input": input2})

    input3 = np.random.rand(1, 5, 5).astype(np.float64)
    generated_inputs.append({"input": input3})

    input4 = np.array([-1, 0, 1]).astype(np.int32)
    generated_inputs.append({"input": input4})

    input5 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    generated_inputs.append({"input": input5})

    input6 = np.array([1+1j, 2+2j, 3+3j]).astype(np.complex128)
    generated_inputs.append({"input": input6})
    
    input7 = np.zeros((2,3)).astype(np.bool_)
    generated_inputs.append({"input": input7})

    return generated_inputs

generated_inputs = rand_like_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rand_like', generated_inputs)
