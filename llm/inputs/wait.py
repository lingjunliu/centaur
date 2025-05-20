
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np
import torch.jit
import threading

def wait_inputs():
    list_of_inputs = []

    # Helper function to create and complete a future
    def create_completed_future(result):
        future = torch.jit.Future()
        future.set_result(result)
        return future

    # Mock torch.jit.Future object with a float tensor
    future1 = create_completed_future(torch.randn(2, 3).float())
    input_dict1 = {"future": future1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Mock torch.jit.Future object with an int tensor
    future2 = create_completed_future(torch.randint(0, 10, (2, 3)))
    input_dict2 = {"future": future2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs = wait_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('wait', generated_inputs)
