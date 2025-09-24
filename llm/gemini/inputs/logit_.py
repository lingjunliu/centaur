
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def logit__inputs():
    list_of_inputs = []

    input1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    eps1 = 1e-6
    input_dict1 = {"input": input1, "eps": eps1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.2, 0.6], [0.4, 0.8]], dtype=np.float64)
    eps2 = 1e-8
    input_dict2 = {"input": input2, "eps": eps2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.01, 0.99, 0.5, 0.2, 0.8], dtype=np.float32)
    eps3 = 1e-5
    input_dict3 = {"input": input3, "eps": eps3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[0.3, 0.7], [0.1, 0.9]], [[0.6, 0.4], [0.8, 0.2]]], dtype=np.float64)
    eps4 = 1e-7
    input_dict4 = {"input": input4, "eps": eps4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    eps5 = 1e-6
    input_dict5 = {"input": input5, "eps": eps5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[0.25, 0.75], [0.33, 0.67]], dtype=np.float64)
    eps6 = 1e-9
    input_dict6 = {"input": input6, "eps": eps6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.123, 0.456, 0.789], dtype=np.float32)
    eps7 = 1e-4
    input_dict7 = {"input": input7, "eps": eps7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = logit__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logit_', generated_inputs)
