
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logsoftmax_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dim1 = 0
    input_dict1 = {"dim": dim1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]], dtype=np.float32)
    dim2 = 1
    input_dict2 = {"dim": dim2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([0.5, -0.2, 1.7, -0.9], dtype=np.float32)
    dim3 = 0
    input_dict3 = {"dim": dim3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(2, 4, 3).astype(np.float32)
    dim4 = 2
    input_dict4 = {"dim": dim4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1e-6, 1e-5, 1e-4]], dtype=np.float32)
    dim5 = 1
    input_dict5 = {"dim": dim5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    dim6 = 1
    input_dict6 = {"dim": dim6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.nn.LogSoftmax"] = logsoftmax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LogSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LogSoftmax'.")


check_valid('torch.nn.LogSoftmax', generated_inputs['torch.nn.LogSoftmax'], lib="torch", suffix=0)
