
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def kthvalue_inputs():
    list_of_inputs = []

    # Case 1: 1D tensor, default dim
    input1 = torch.arange(1., 6.).numpy()
    k1 = 4
    input_dict1 = {
        "input": input1,
        "k": k1,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.kthvalue"] = kthvalue_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.kthvalue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.kthvalue'.")

check_valid('torch.kthvalue', generated_inputs['torch.kthvalue'], lib="torch")
