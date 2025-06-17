
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def msort_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D integer tensor
    input2 = torch.randint(-5, 5, (5, 4)).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.msort"] = msort_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.msort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.msort'.")

check_valid('torch.msort', generated_inputs['torch.msort'], lib="torch")
