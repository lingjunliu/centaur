
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def adaptiveavgpool1d_inputs_2():
    list_of_inputs = []

    input_arr = torch.randn(1, 64, 8).numpy()
    input_dict = {"output_size": (5,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones((10, 3, 15), dtype=torch.float32).numpy()
    input_dict = {"output_size": (1,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(-1, 1, steps=60).reshape(3, 20).numpy()
    input_dict = {"output_size": (3,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(-14, 0, dtype=torch.float32).reshape(2, 1, 7).numpy()
    input_dict = {"output_size": (7,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(4, 16) * 10.0).numpy()
    input_dict = {"output_size": (4,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 5).numpy()
    input_dict = {"output_size": (9,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 2).numpy()
    input_dict = {"output_size": (2,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 5, 30, dtype=torch.float64).numpy()
    input_dict = {"output_size": (15,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy()
    input_dict = {"output_size": (10,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 5, dtype=torch.float16).numpy()
    input_dict = {"output_size": (5,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.linspace(0, 1, steps=13, dtype=torch.float64).reshape(1, 13).numpy()
    input_dict = {"output_size": (6,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.arange(84, dtype=torch.float32).reshape(4, 7, 3).numpy()
    input_dict = {"output_size": (3,), "input": input_arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveAvgPool1d_2"] = adaptiveavgpool1d_inputs_2()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveAvgPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool1d_2'.")


check_valid('torch.nn.AdaptiveAvgPool1d', generated_inputs['torch.nn.AdaptiveAvgPool1d_2'], lib="torch", suffix=2)
