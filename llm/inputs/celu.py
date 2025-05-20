
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []

    input1 = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    alpha1 = 1.0
    input_dict1 = {"input": input1, "alpha": alpha1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-1, -0.5], [0, 0.5]], dtype=np.float64)
    alpha2 = 0.5
    input_dict2 = {"input": input2, "alpha": alpha2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-1, -0.5], [0, 0.5]], dtype=np.float32)
    alpha3 = 2.0
    input_dict3 = {"input": input3, "alpha": alpha3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1, -0.5], [0, 0.5]], dtype=np.float16)
    alpha4 = 0.75
    input_dict4 = {"input": input4, "alpha": alpha4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    alpha5 = 1.5
    input_dict5 = {"input": input5, "alpha": alpha5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)
    alpha6 = 0.25
    input_dict6 = {"input": input6, "alpha": alpha6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.float64)
    alpha7 = 0.8
    input_dict7 = {"input": input7, "alpha": alpha7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    alpha8 = 0.9
    input_dict8 = {"input": input8, "alpha": alpha8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = celu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('celu', generated_inputs)
