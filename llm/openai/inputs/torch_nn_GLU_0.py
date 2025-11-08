
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def glu_inputs():
    list_of_inputs = []

    input_arr = torch.randn(8).numpy()
    input_dict = {"dim": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 10).numpy()
    input_dict = {"dim": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(10, 6, dtype=torch.float64).numpy()
    input_dict = {"dim": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 8).numpy()
    input_dict = {"dim": -1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 12, 7).numpy()
    input_dict = {"dim": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 5, 6, 4).numpy()
    input_dict = {"dim": -2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 2, 7, 10).numpy()
    input_dict = {"dim": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 2, 2, 14).numpy()
    input_dict = {"dim": -1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.randn(6, 8)
    input_arr = base.t().numpy()
    input_dict = {"dim": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 4, 4, dtype=torch.float16).numpy()
    input_dict = {"dim": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 64, 3).numpy()
    input_dict = {"dim": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(7, 2, 2).numpy()
    input_dict = {"dim": -2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.GLU"] = glu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GLU'.")


check_valid('torch.nn.GLU', generated_inputs['torch.nn.GLU'], lib="torch", suffix=0)
