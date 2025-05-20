
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def det_inputs():
    list_of_inputs = []

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.eye(3)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 2], [3, 4]], dtype=np.int64)
    A = A.astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 2, 2)
    input_dict = {"A": A[0]}
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
