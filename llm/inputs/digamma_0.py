
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def digamma_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input1 = np.array(2.5, dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D tensor with positive floats
    input2 = np.array([1.0, 2.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.digamma"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.digamma'.")

check_valid('torch.digamma', generated_inputs['torch.digamma'], lib="torch")
