
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def conv_transpose2d_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(1, 1, 3, 3, dtype=torch.float32).numpy()
    weight = torch.randn(1, 1, 3, 3, dtype=torch.float32).numpy()
    bias = torch.randn(1, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(1, 2, 5, 5, dtype=torch.float32).numpy()
    weight = torch.randn(2, 4, 2, 2, dtype=torch.float32).numpy()
    bias = torch.randn(4, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 2, "padding": 1, "output_padding": 1, "groups": 1, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(2, 4, 4, 4, dtype=torch.float32).numpy()
    weight = torch.randn(4, 3, 3, 3, dtype=torch.float32).numpy()
    bias = torch.randn(6, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 2, "padding": 0, "output_padding": 0, "groups": 2, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = torch.randn(1, 3, 6, 6, dtype=torch.float32).numpy()
    weight = torch.randn(3, 1, 3, 3, dtype=torch.float32).numpy()
    bias = torch.randn(3, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 1, "padding": 2, "output_padding": 0, "groups": 3, "dilation": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = torch.tensor([[[[-1.0, 2.0],
                            [3.0, -4.0]]]], dtype=torch.float32).numpy()
    weight = torch.tensor([[[[1.0]]],
                           [[[ -0.5]]]], dtype=torch.float32).permute(1,0,2,3).numpy()  # shape (1,2,1,1)
    bias = torch.tensor([-1.0, 0.5], dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 3, "padding": 0, "output_padding": 2, "groups": 1, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = torch.randn(3, 2, 4, 5, dtype=torch.float32).numpy()
    weight = torch.randn(2, 2, 2, 3, dtype=torch.float32).numpy()
    bias = torch.randn(2, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 2, "padding": 1, "output_padding": 1, "groups": 1, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = torch.randn(1, 4, 3, 3, dtype=torch.float32).numpy()
    weight = torch.randn(4, 2, 2, 2, dtype=torch.float32).numpy()
    bias = torch.randn(8, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 2, "padding": 1, "output_padding": 0, "groups": 4, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = torch.randn(2, 2, 5, 5, dtype=torch.float32).numpy()
    weight = torch.randn(2, 4, 3, 3, dtype=torch.float32).numpy()
    bias = torch.randn(4, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 2, "padding": 2, "output_padding": 1, "groups": 1, "dilation": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = torch.randn(1, 5, 2, 3, dtype=torch.float64).numpy()
    weight = torch.randn(5, 1, 2, 2, dtype=torch.float64).numpy()
    bias = torch.randn(5, dtype=torch.float64).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 1, "padding": 0, "output_padding": 0, "groups": 5, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = torch.ones(1, 1, 1, 1, dtype=torch.float32).numpy()
    weight = torch.ones(1, 1, 1, 1, dtype=torch.float32).numpy()
    bias = torch.zeros(1, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input = torch.randn(1, 2, 3, 3, dtype=torch.float32).numpy()
    weight = torch.randn(2, 3, 4, 4, dtype=torch.float32).numpy()
    bias = torch.randn(6, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 3, "padding": 1, "output_padding": 1, "groups": 2, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input = torch.randn(2, 6, 4, 4, dtype=torch.float32).numpy()
    weight = torch.randn(6, 3, 1, 5, dtype=torch.float32).numpy()
    bias = torch.randn(9, dtype=torch.float32).numpy()
    input_dict = {"input": input, "weight": weight, "bias": bias, "stride": 2, "padding": 0, "output_padding": 1, "groups": 3, "dilation": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.conv_transpose2d_1"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.conv_transpose2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose2d_1'.")


check_valid('torch.nn.functional.conv_transpose2d', generated_inputs['torch.nn.functional.conv_transpose2d_1'], lib="torch", suffix=1)
