
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def replicationpad1d_inputs():
    list_of_inputs = []

    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 5, dtype=torch.float64).numpy()
    input_dict = {"padding": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.tensor([[7]], dtype=torch.int64).numpy()
    input_dict = {"padding": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(-50, 50, (4, 1, 3), dtype=torch.int8).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 10, dtype=torch.float32).numpy()
    input_dict = {"padding": 5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 7, dtype=torch.float32).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 256, (5, 2), dtype=torch.uint8).numpy()
    input_dict = {"padding": 4, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(0, 1, steps=10, dtype=torch.float64).reshape(10, 1).numpy()
    input_dict = {"padding": 6, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(50, dtype=torch.int32).reshape(1, 1, 50).numpy()
    input_dict = {"padding": 10, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 100, dtype=torch.float32).numpy()
    input_dict = {"padding": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.rand(5, 4, 4, dtype=torch.float32).numpy()
    input_dict = {"padding": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad1d_1"] = replicationpad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad1d_1'.")


check_valid('torch.nn.ReplicationPad1d', generated_inputs['torch.nn.ReplicationPad1d_1'], lib="torch", suffix=1)
