
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def leaky_relu_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.tensor([-3.0, -1.0, 0.0, 1.5, 3.0], dtype=torch.float32).numpy()
    negative_slope = 0.01
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 2
    input = torch.tensor([[-1.0, 0.0, 1.0],
                          [2.0, -2.5, 3.5]], dtype=torch.float64).numpy()
    negative_slope = np.float64(0.2)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 3
    input = torch.tensor([[[-1.0, 2.0, -3.0],
                           [4.0, -5.0, 6.0],
                           [0.0, 7.0, -8.0]]], dtype=torch.float32).numpy()
    negative_slope = 2.0
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 4
    input = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    negative_slope = np.float32(0.05)
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 5
    input = torch.tensor(-3.5, dtype=torch.float32).numpy()
    negative_slope = 0.5
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 6
    input = torch.tensor([-1000.0, -0.5, 0.5, 1000.0], dtype=torch.float16).numpy()
    negative_slope = np.float16(0.01)
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 7
    input = (torch.ones(1, 2, 1, 2, 3, dtype=torch.float32) * -1.5).numpy()
    negative_slope = 1.0
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 8
    input = torch.tensor([[-1.2, 3.4],
                          [0.0, -0.0]], dtype=torch.float32).numpy()
    negative_slope = 0.0
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 9
    input = torch.empty(0, 3, dtype=torch.float32).numpy()
    negative_slope = np.float64(0.3)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 10
    input = torch.empty(3, 0, 2, dtype=torch.float32).numpy()
    negative_slope = -0.5
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 11
    input = torch.tensor([np.nan, np.inf, -np.inf, -1.0, 1.0], dtype=torch.float32).numpy()
    negative_slope = np.float32(0.1)
    inplace = False
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    # Input 12
    input = torch.arange(24, dtype=torch.float32).view(2, 3, 4).transpose(1, 2).numpy()
    negative_slope = 0.8
    inplace = True
    list_of_inputs.append(copy.deepcopy({"input": input, "negative_slope": negative_slope, "inplace": inplace}))

    return list_of_inputs

generated_inputs["torch.nn.functional.leaky_relu"] = leaky_relu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.leaky_relu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.leaky_relu'.")


check_valid('torch.nn.functional.leaky_relu', generated_inputs['torch.nn.functional.leaky_relu'], lib="torch", suffix=0)
