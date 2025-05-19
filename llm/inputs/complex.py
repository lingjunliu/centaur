
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def complex_inputs():
    list_of_inputs = []

    real1 = np.array([1, 2, 3], dtype=np.float32)
    imag1 = np.array([4, 5, 6], dtype=np.float32)
    input_dict1 = {"real": real1, "imag": imag1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    real2 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    imag2 = np.array([[5, 6], [7, 8]], dtype=np.float64)
    input_dict2 = {"real": real2, "imag": imag2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    real3 = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float32)
    imag3 = np.array([[5.5, -6.5], [-7.5, 8.5]], dtype=np.float32)
    input_dict3 = {"real": real3, "imag": imag3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    real4 = np.array([1, 2, 3], dtype=np.float64)
    imag4 = np.array([4, 5, 6], dtype=np.float64)
    input_dict4 = {"real": real4, "imag": imag4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    real5 = np.array([1], dtype=np.float32)
    imag5 = np.array([0], dtype=np.float32)
    input_dict5 = {"real": real5, "imag": imag5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    real6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    imag6 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float64)
    input_dict6 = {"real": real6, "imag": imag6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    real7 = np.array([], dtype=np.float32)
    imag7 = np.array([], dtype=np.float32)
    input_dict7 = {"real": real7, "imag": imag7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = complex_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('complex', generated_inputs)
