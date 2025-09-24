
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def is_inference_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float tensor
    tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor1}))

    # Input 2: 2D integer tensor
    tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor2}))

    # Input 3: 3D double tensor
    tensor3 = torch.randn(2, 3, 4, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor3}))

    # Input 4: Tensor with negative values
    tensor4 = torch.tensor([-1.5, -0.5, 0.0, 1.5, 2.5]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor4}))

    # Input 5: Scalar tensor (0-dimensional)
    tensor5 = torch.tensor(42).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor5}))

    # Input 6: Single-element tensor
    tensor6 = torch.tensor([100.0]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor6}))

    # Input 7: Boolean tensor
    tensor7 = torch.tensor([[True, False], [False, True]]).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor7}))

    # Input 8: Empty tensor (one dimension is 0)
    tensor8 = torch.empty((2, 0, 3)).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor8}))

    # Input 9: Complex tensor
    tensor9 = torch.tensor([1+2j, 3-4j, -5j], dtype=torch.complex64).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor9}))

    # Input 10: Tensor that was created with requires_grad=True, then detached
    tensor10 = torch.zeros(5, requires_grad=True).detach().numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor10}))

    # Input 11: 4D tensor
    tensor11 = torch.ones(2, 2, 2, 2, dtype=torch.int8).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor11}))
    
    # Input 12: Tensor with a large number of elements
    tensor12 = torch.linspace(0, 100, steps=1000).numpy()
    list_of_inputs.append(copy.deepcopy({"input": tensor12}))

    return list_of_inputs

generated_inputs["torch.is_inference"] = is_inference_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_inference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_inference'.")

check_valid('torch.is_inference', generated_inputs['torch.is_inference'], lib="torch", suffix=0)
