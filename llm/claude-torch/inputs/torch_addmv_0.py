
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def addmv_inputs():
    list_of_inputs = []
    
    input_vec = torch.randn(2).numpy()
    mat = torch.randn(2, 3).numpy()
    vec = torch.randn(3).numpy()
    beta = 1.0
    alpha = 1.0
    out = torch.empty(2).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_vec = torch.ones(4).numpy()
    mat = torch.randn(4, 5).numpy()
    vec = torch.randn(5).numpy()
    beta = 0.5
    alpha = 2.0
    out = torch.empty(4).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_vec = torch.randn(3).numpy()
    mat = torch.randn(3, 2).numpy()
    vec = torch.randn(2).numpy()
    beta = 0.0
    alpha = 1.0
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_vec = torch.randn(5).numpy()
    mat = torch.randn(5, 4).numpy()
    vec = torch.randn(4).numpy()
    beta = -1.0
    alpha = -0.5
    out = torch.empty(5).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_vec = torch.randn(10).numpy()
    mat = torch.randn(10, 20).numpy()
    vec = torch.randn(20).numpy()
    beta = 1.0
    alpha = 1.0
    out = torch.empty(10).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_vec = torch.randn(1).numpy()
    mat = torch.randn(1, 1).numpy()
    vec = torch.randn(1).numpy()
    beta = 1.0
    alpha = 1.0
    out = torch.empty(1).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        "alpha": alpha,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_vec = torch.randn(3).numpy()
    mat = torch.randn(3, 7).numpy()
    vec = torch.randn(7).numpy()
    beta = 1.0
    alpha = 0.0
    out = torch.empty(3).numpy()
    input_dict = {
        "input": input_vec,
        "mat": mat,
        "vec": vec,
        "beta": beta,
        

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.addmv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addmv'.")


check_valid('torch.addmv', generated_inputs['torch.addmv'], lib="torch", suffix=0)
