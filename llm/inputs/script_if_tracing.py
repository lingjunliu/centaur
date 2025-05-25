
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def script_if_tracing_inputs():
    list_of_inputs = []

    # Input 1: Condition True, fn and alternative_fn return different values
    input_dict = {
        "condition": True,
        "fn": [torch.randn(1, 3, 5, 5).numpy(), torch.randint(0, 10, (2, 2)).numpy()],
        "alternative_fn": [torch.zeros(1, 3, 5, 5).numpy(), np.array([1, 2, 3])]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = script_if_tracing_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('script_if_tracing', generated_inputs)
