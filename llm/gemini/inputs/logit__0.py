
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logit__inputs():
    list_of_inputs = []

    input1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    eps1 = 1e-6
    input_dict1 = {"input": input1, "eps": eps1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float64)
    eps2 = 1e-7
    input_dict2 = {"input": input2, "eps": eps2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[0.01, 0.99], [0.2, 0.8]], [[0.3, 0.7], [0.4, 0.6]]], dtype=np.float32)
    eps3 = 1e-5
    input_dict3 = {"input": input3, "eps": eps3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([0.001, 0.999, 0.5, 0.25, 0.75], dtype=np.float16)
    eps4 = 1e-4
    input_dict4 = {"input": input4, "eps": eps4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    eps5 = 0.0
    input_dict5 = {"input": input5, "eps": eps5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float64)
    eps6 = 0.01
    input_dict6 = {"input": input6, "eps": eps6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([0.5], dtype=np.float32)
    eps7 = 1e-8
    input_dict7 = {"input": input7, "eps": eps7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.logit_"] = logit__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logit_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logit_'.")

check_valid('torch.logit_', generated_inputs['torch.logit_'], lib="torch")
