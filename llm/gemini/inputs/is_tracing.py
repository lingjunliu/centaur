
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def is_tracing_inputs():
    list_of_inputs = []

    input_dict = {"input": np.array([1])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([1.0])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([[1, 2], [3, 4]])}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {"input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {"input": np.array([-1, -2, -3])}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.is_tracing"] = is_tracing_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('is_tracing', generated_inputs['torch.jit.is_tracing'], lib="torch")
