
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def linear_inputs():
    list_of_inputs = []

    # 1
    input = torch.randn(4, 3, dtype=torch.float32).numpy()
    weight = torch.randn(5, 3, dtype=torch.float32).numpy()
    bias = torch.randn(5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 2
    input = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32).numpy()
    weight = torch.tensor([[0.5, -1.0, 2.0], [-0.3, 0.7, -0.8]], dtype=torch.float32).numpy()
    bias = torch.tensor([-0.1, 0.2], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 3
    input = torch.randn(2, 4, 3, dtype=torch.float32).numpy()
    weight = torch.randn(6, 3, dtype=torch.float32).numpy()
    bias = torch.randn(6, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 4
    input = torch.randn(1, 2, 3, 4, dtype=torch.float64).numpy()
    weight = torch.randn(7, 4, dtype=torch.float64).numpy()
    bias = torch.randn(7, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 5 (out_features=1)
    input = torch.randn(5, 3, dtype=torch.float32).numpy()
    weight = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float32).numpy()
    bias = torch.tensor([0.75], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 6 (1D input -> vector output)
    input = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float32).numpy()
    weight = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32).numpy()
    bias = torch.tensor([ -0.5 ], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 7 (larger batch)
    input = torch.randn(10, 10, dtype=torch.float32).numpy()
    weight = torch.randn(8, 10, dtype=torch.float32).numpy()
    bias = torch.randn(8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 8 (float64 larger features)
    input = torch.randn(8, 16, dtype=torch.float64).numpy()
    weight = torch.randn(32, 16, dtype=torch.float64).numpy()
    bias = torch.randn(32, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 9 (empty out_features)
    input = torch.randn(2, 3, dtype=torch.float32).numpy()
    weight = torch.empty(0, 3, dtype=torch.float32).numpy()
    bias = torch.empty(0, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 10 (zero-sized batch)
    input = torch.empty(0, 4, dtype=torch.float32).numpy()
    weight = torch.randn(3, 4, dtype=torch.float32).numpy()
    bias = torch.randn(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 11 (non-contiguous arrays via striding)
    base_in = torch.arange(6 * 4, dtype=torch.float32).view(6, 4).numpy()
    input = base_in[::2]
    base_w = torch.arange(5 * 4, dtype=torch.float32).view(4, 5).numpy()
    weight = base_w.T
    bias = torch.tensor([-1.0, 0.0, 1.0, -2.0, 2.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 12 (special values in input)
    input = np.array([[np.nan, 1.0, -2.0, np.inf],
                      [-np.inf, 0.0, 3.5, -4.5]], dtype=np.float32)
    weight = torch.tensor([[0.25, -0.5, 0.75, 1.0]], dtype=torch.float32).numpy()
    bias = torch.tensor([0.1], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    # 13 (zero in_features)
    input = torch.empty(2, 0, dtype=torch.float32).numpy()
    weight = torch.empty(3, 0, dtype=torch.float32).numpy()
    bias = torch.randn(3, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": input, "weight": weight, "bias": bias}))

    return list_of_inputs

generated_inputs["torch.nn.functional.linear"] = linear_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.linear' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.linear'.")


check_valid('torch.nn.functional.linear', generated_inputs['torch.nn.functional.linear'], lib="torch", suffix=0)
