
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def eye_inputs():
    list_of_inputs = []

    # Input 1
    n = 3
    m = 3
    out = torch.empty(3, 3)
    dtype = torch.float32
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    n = 5
    m = None
    out = torch.empty(5, 5)
    dtype = torch.float64
    layout = "strided"
    requires_grad = True
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    n = 2
    m = 4
    out = torch.empty(2, 4)
    dtype = torch.int64
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    n = 4
    m = 2
    out = torch.empty(4, 2)
    dtype = torch.int32
    layout = "strided"
    requires_grad = True
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    n = 1
    m = 1
    out = torch.empty(1, 1)
    dtype = torch.float16
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    n = 6
    m = None
    out = torch.empty(6, 6)
    dtype = torch.bfloat16
    layout = "strided"
    requires_grad = True
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    n = 7
    m = 8
    out = torch.empty(7, 8)
    dtype = torch.uint8
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    n = 8
    m = 7
    out = torch.empty(8, 7)
    dtype = torch.int8
    layout = "strided"
    requires_grad = True
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    n = 9
    m = None
    out = torch.empty(9, 9)
    dtype = torch.bool
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    n = 10
    m = 5
    out = torch.empty(10, 5)
    dtype = torch.float32
    layout = "strided"
    requires_grad = True
    input_dict = {"n": n, "m": m, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Only n, with out
    n = 4
    out = torch.empty(4, 4)
    dtype = torch.float32
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": None, "out": out, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12, no out
    n = 2
    m = 3
    dtype = torch.float32
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": m, "out": None, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    #Input 13, No m, no out
    n = 4
    dtype = torch.float32
    layout = "strided"
    requires_grad = False
    input_dict = {"n": n, "m": None, "out": None, "dtype": dtype, "layout": layout, "requires_grad": requires_grad}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.eye"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.eye' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.eye'.")

check_valid('torch.eye', generated_inputs['torch.eye'], lib="torch", suffix=0)
