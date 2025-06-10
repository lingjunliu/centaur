
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def gammainc_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    x = np.array([0.5, 1.5, 2.5]).astype(np.float32)
    input_dict = {"a": a, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.special.gammainc"] = gammainc_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('gammainc', generated_inputs['torch.special.gammainc'], lib="torch")
