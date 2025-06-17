
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def log_softmax_inputs():
    list_of_inputs = []

    input_1 = np.random.randn(3, 5).astype(np.float32)
    dim_1 = 1
    input_dict_1 = {"input": input_1, "dim": dim_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))


    return list_of_inputs

generated_inputs["torch.nn.functional.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.log_softmax_2'.")

check_valid('torch.nn.functional.log_softmax', generated_inputs['torch.nn.functional.log_softmax_2'], lib="torch")
