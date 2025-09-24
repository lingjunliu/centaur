
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def LogSigmoid_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(1, 1, 1, 1).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([-1, 0, 1]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(5).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = LogSigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LogSigmoid', generated_inputs['torch.nn.LogSigmoid'], lib="torch")
