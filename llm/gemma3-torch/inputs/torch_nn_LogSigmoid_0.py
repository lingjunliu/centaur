
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logsigmoid_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": torch.tensor(input1)}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict2 = {"input": torch.tensor(input2)}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.LogSigmoid"] = logsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LogSigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LogSigmoid'.")


check_valid('torch.nn.LogSigmoid', generated_inputs['torch.nn.LogSigmoid'], lib="torch", suffix=0)
