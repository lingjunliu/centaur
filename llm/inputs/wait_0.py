
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy

def jit_wait_inputs():
    list_of_inputs = []

    # Minimal example: Create a real torch._C.Future
    fut = torch.jit.fork(lambda x: x + 1, torch.tensor(1))
    input_dict1 = {"future": fut}
    list_of_inputs.append(input_dict1)
    
    fut2 = torch.jit.fork(lambda x: x * 2, torch.tensor(2.0))
    input_dict2 = {"future": fut2}
    list_of_inputs.append(input_dict2)

    fut3 = torch.jit.fork(lambda x: x.sum(), torch.randn(3,3))
    input_dict3 = {"future": fut3}
    list_of_inputs.append(input_dict3)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.jit.wait"] = jit_wait_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.jit.wait' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.wait'.")

check_valid('torch.jit.wait', generated_inputs['torch.jit.wait'], lib="torch")
