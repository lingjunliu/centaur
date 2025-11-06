
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def zero_pad2d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 1, 3, 3).numpy()
    input_dict = {"padding": 0, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones((2, 3, 4, 5)).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 3, 3).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 2, 2, dtype=torch.float64).numpy()
    input_dict = {"padding": 3, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 100, (1, 2, 5, 5), dtype=torch.int32).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.rand(3, 4, 4) > 0.5).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 1, 0, 5).numpy()
    input_dict = {"padding": 2, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 0).numpy()
    input_dict = {"padding": 1, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 4, 8, 8, dtype=torch.float16).numpy()
    input_dict = {"padding": 4, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(-10, 10, (3, 5, 7, 9), dtype=torch.int64).numpy()
    input_dict = {"padding": 5, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones(1, 1, 1).numpy()
    input_dict = {"padding": 10, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randint(0, 256, (1, 1, 2, 3), dtype=torch.uint8).numpy()
    input_dict = {"padding": 4, "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ZeroPad2d_1"] = zero_pad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ZeroPad2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad2d_1'.")


check_valid('torch.nn.ZeroPad2d', generated_inputs['torch.nn.ZeroPad2d_1'], lib="torch", suffix=1)
