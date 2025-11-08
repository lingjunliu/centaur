
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def adaptiveavgpool2d_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(1, 64, 8, 9, dtype=torch.float32).numpy()
    output_size = (5, 7)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 2
    input = torch.randn(1, 64, 10, 9, dtype=torch.float32).numpy()
    output_size = (7, 7)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 3 (no batch dim)
    input = torch.randn(3, 32, 32, dtype=torch.float32).numpy()
    output_size = (8, 8)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 4 (global avg pool)
    input = torch.randn(2, 3, 224, 224, dtype=torch.float32).numpy()
    output_size = (1, 1)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 5 (output equals input size)
    input = torch.randn(4, 1, 7, 5, dtype=torch.float32).numpy()
    output_size = (7, 5)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 6 (float64)
    input = torch.randn(1, 128, 16, 20, dtype=torch.float64).numpy()
    output_size = (4, 4)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 7 (no batch dim, rectangular output)
    input = torch.randn(5, 15, 17, dtype=torch.float32).numpy()
    output_size = (5, 1)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 8 (constructed with negative to positive values)
    input = torch.linspace(-5.0, 5.0, steps=3 * 2 * 6 * 6, dtype=torch.float32).reshape(3, 2, 6, 6).numpy()
    output_size = (2, 3)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 9 (more channels)
    input = torch.randn(8, 16, 13, 11, dtype=torch.float32).numpy()
    output_size = (3, 4)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    # Input 10 (downsample to smaller non-square)
    input = torch.randn(2, 5, 9, 7, dtype=torch.float32).numpy()
    output_size = (3, 2)
    list_of_inputs.append(copy.deepcopy({"output_size": output_size, "input": input}))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveAvgPool2d_2"] = adaptiveavgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveAvgPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveAvgPool2d_2'.")


check_valid('torch.nn.AdaptiveAvgPool2d', generated_inputs['torch.nn.AdaptiveAvgPool2d_2'], lib="torch", suffix=2)
