
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_nonzero_inputs():
    list_of_inputs = []

    input = torch.tensor(True, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(False, dtype=torch.bool).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-5, dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0, dtype=torch.int16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(123456, dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-1234567890123, dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0, dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(200, dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(0.0, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(-3.14, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(2.718281828, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    input = torch.tensor(float('inf'), dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input}))

    return list_of_inputs

generated_inputs["torch.is_nonzero"] = is_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_nonzero'.")


check_valid('torch.is_nonzero', generated_inputs['torch.is_nonzero'], lib="torch", suffix=0)
