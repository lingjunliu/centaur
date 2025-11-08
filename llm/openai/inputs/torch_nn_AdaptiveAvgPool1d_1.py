
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def adaptiveavgpool1d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 64, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 5, "input": input_arr}))

    input_arr = torch.randn(64, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 5, "input": input_arr}))

    input_arr = torch.randn(2, 3, 15, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 7, "input": input_arr}))

    input_arr = torch.arange(4 * 1 * 10, dtype=torch.float16).reshape(4, 1, 10).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 3, "input": input_arr}))

    input_arr = torch.tensor([[[0.5]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 1, "input": input_arr}))

    input_arr = torch.linspace(-3.5, 3.5, steps=3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 14, "input": input_arr}))

    input_arr = torch.ones(5, 2, 4, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 4, "input": input_arr}))

    input_arr = (torch.randn(2, 4, 9, dtype=torch.float32) * 1000.0).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 1, "input": input_arr}))

    input_arr = torch.arange(3 * 50, dtype=torch.float64).reshape(3, 50).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 25, "input": input_arr}))

    input_arr = torch.randn(8, 3, 2, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 6, "input": input_arr}))

    t = torch.linspace(0, 20, steps=1 * 3 * 1000, dtype=torch.float32).reshape(1, 3, 1000)
    input_arr = torch.sin(t).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 10, "input": input_arr}))

    input_arr = torch.linspace(-5, 5, steps=13, dtype=torch.float64).reshape(1, 13).numpy()
    list_of_inputs.append(copy.deepcopy({"output_size": 7, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveAvgPool1d_1"] = adaptiveavgpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveAvgPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool1d_1'.")


check_valid('torch.nn.AdaptiveAvgPool1d', generated_inputs['torch.nn.AdaptiveAvgPool1d_1'], lib="torch", suffix=1)
