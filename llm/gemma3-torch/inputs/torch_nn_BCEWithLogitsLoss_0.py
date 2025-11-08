
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bcewithlogitsloss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(10, 64).astype(np.float32)
    target1 = np.random.randint(0, 2, size=(10, 64)).astype(np.float32)
    pos_weight1 = np.ones(64).astype(np.float32)
    
    input_dict1 = {
        "weight": np.array([1.0]).astype(np.float32),
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "pos_weight": pos_weight1,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    return list_of_inputs

generated_inputs["torch.nn.BCEWithLogitsLoss"] = bcewithlogitsloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BCEWithLogitsLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCEWithLogitsLoss'.")


check_valid('torch.nn.BCEWithLogitsLoss', generated_inputs['torch.nn.BCEWithLogitsLoss'], lib="torch", suffix=0)
