
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def pdist_inputs():
    list_of_inputs = []

    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    p1 = 2.0
    input_dict1 = {"input": input1, "p": p1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = pdist_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('pdist', generated_inputs)
