
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def erfc_inputs():
    list_of_inputs = []

    # Input 1: Scalar float
    input1 = np.array(1.0, dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.erfc"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.erfc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.erfc'.")

check_valid('torch.erfc', generated_inputs['torch.erfc'], lib="torch")
