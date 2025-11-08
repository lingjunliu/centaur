
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def zero_pad2d_inputs():
    list_of_inputs = []

    # Input 1
    padding = (2, 2, 2, 2)
    input_arr = torch.randn(1, 1, 3, 3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 2
    padding = (1, 1, 0, 3)
    input_arr = torch.randn(3, 4, 5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 3
    padding = (0, 0, 0, 0)
    input_arr = torch.randn(2, 3, 4, 4, dtype=torch.float16).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 4
    padding = (3, 1, 2, 0)
    input_arr = torch.randint(-100, 100, (5, 2, 6, 7), dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 5
    padding = (1, 0, 1, 0)
    input_arr = (torch.randint(0, 2, (3, 5, 5)) > 0).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 6
    padding = (5, 0, 0, 5)
    input_arr = torch.randint(0, 256, (2, 3, 1, 2), dtype=torch.uint8).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 7
    padding = (2, 4, 1, 3)
    input_arr = torch.randint(-128, 128, (8, 9, 10), dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 8
    padding = (1, 2, 3, 4)
    input_arr = torch.randn(2, 4, 5, 6, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 9
    padding = (10, 0, 0, 0)
    input_arr = torch.ones(1, 2, 1, 1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 10
    padding = (0, 3, 2, 2)
    input_arr = torch.randn(2, 3, 16, 16, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 11
    padding = (7, 7, 7, 7)
    input_arr = torch.randint(-1000, 1000, (1, 1, 10, 10), dtype=torch.int64).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    # Input 12
    padding = (1, 1, 1, 1)
    input_arr = torch.tensor([[[[float('nan'), -1.0, 2.0],
                                [3.0, float('inf'), -float('inf')]]]], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"padding": padding, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.ZeroPad2d_2"] = zero_pad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ZeroPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ZeroPad2d_2'.")


check_valid('torch.nn.ZeroPad2d', generated_inputs['torch.nn.ZeroPad2d_2'], lib="torch", suffix=2)
