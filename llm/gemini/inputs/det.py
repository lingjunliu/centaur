
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def det_inputs():
    list_of_inputs = []

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.array([[1.5, 2.5], [3.5, 4.5]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(4, 4)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = det_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('det', generated_inputs)
