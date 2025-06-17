
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_clip_inputs():
    list_of_inputs = []

    input_tensor = np.random.randn(3, 4, 5).astype(np.float32)
    min_val = -0.5
    max_val = 1.0
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.random.randint(-5, 5, size=(2, 2), dtype=np.int32)
    min_val = 0
    max_val = 3
    input_dict = {"input": input_tensor, "min": float(min_val), "max": float(max_val)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).astype(np.float64)
    min_val = -0.5
    max_val = 2.5
    input_dict = {"input": input_tensor, "min": min_val, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.clip"] = torch_clip_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.clip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.clip'.")

check_valid('torch.clip', generated_inputs['torch.clip'], lib="torch")
