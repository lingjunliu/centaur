
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def randint_like_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([1, 2, 3])
    high = 5
    dtype = np.int64
    layout = 'strided'
    requires_grad = False
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.ones((2, 3))
    high = 10
    dtype = np.int64
    layout = 'strided'
    requires_grad = True
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.zeros((1, 4, 2))
    high = 2
    dtype = np.int32
    layout = 'strided'
    requires_grad = False
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.random.rand(5, 5).astype(np.float32)
    high = 100
    dtype = np.int64
    layout = 'strided'
    requires_grad = True
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([[-1, -2], [-3, -4]])
    high = 5
    dtype = np.int64
    layout = 'strided'
    requires_grad = False
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    high = 5
    dtype = np.int64
    layout = 'strided'
    requires_grad = True
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.random.randint(0, 10, size=(3, 2, 4)).astype(np.int32)
    high = 20
    dtype = np.int64
    layout = 'strided'
    requires_grad = False
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.array([1, 2, 3], dtype=np.int8)
    high = 10
    dtype = np.int64
    layout = 'strided'
    requires_grad = True
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.ones((2,2), dtype=np.float16)
    high = 5
    dtype = np.int64
    layout = 'strided'
    requires_grad = False
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.zeros((1,5), dtype=np.uint8)
    high = 255
    dtype = np.int64
    layout = 'strided'
    requires_grad = True
    input_dict = {"input": input, "high": high, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.randint_like_2"] = randint_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.randint_like_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.randint_like_2'.")

check_valid('torch.randint_like', generated_inputs['torch.randint_like_2'], lib="torch", suffix=2)
