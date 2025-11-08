
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def replicationpad1d_inputs():
    list_of_inputs = []

    input_arr = torch.arange(8, dtype=torch.float32).reshape(2, 4).numpy()
    padding = (2, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).numpy()
    padding = (3, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.tensor([[0], [1], [2]], dtype=torch.int64).numpy()
    padding = (1, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.randint(-10, 10, (4, 3, 2), dtype=torch.int32).numpy()
    padding = (0, 3)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.arange(5, dtype=torch.float64).reshape(1, 5).numpy()
    padding = (5, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1).numpy()
    padding = (10, 10)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.randn((8, 3), dtype=torch.float32).numpy()
    padding = (0, 0)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32).numpy()
    padding = (2, 2)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.linspace(0, 1, steps=50, dtype=torch.float32).reshape(5, 10).numpy()
    padding = (7, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.arange(63, dtype=torch.float32).reshape(3, 3, 7).numpy()
    padding = (1, 4)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.arange(100, dtype=torch.int64).reshape(10, 10, 1).numpy()
    padding = (1, 1)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    input_arr = torch.tensor([[42]], dtype=torch.int64).numpy()
    padding = (0, 5)
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad1d_2"] = replicationpad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad1d_2'.")


check_valid('torch.nn.ReplicationPad1d', generated_inputs['torch.nn.ReplicationPad1d_2'], lib="torch", suffix=2)
