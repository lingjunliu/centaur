
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def nansum_inputs():
    list_of_inputs = []
    
    # Input 1 - basic tensor with NaN
    input = torch.tensor([1.0, 2.0, float('nan'), 4.0]).numpy()
    dim = None
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - 2D tensor with NaN
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
    dim = None
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - 2D tensor with NaN and dim=0
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
    dim = 0
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - 2D tensor with NaN and dim=1
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
    dim = 1
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - 2D tensor with NaN and keepdim=True
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
    dim = 0
    keepdim = True
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - 2D tensor with NaN and keepdim=True
    input = torch.tensor([[1, 2], [3., float("nan")]]).numpy()
    dim = 1
    keepdim = True
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - 3D tensor with NaN
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
    dim = None
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - 3D tensor with NaN and dim=0
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
    dim = 0
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - 3D tensor with NaN and dim=1
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
    dim = 1
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - 3D tensor with NaN and dim=2
    input = torch.tensor([[[1, 2], [3., float("nan")]], [[4, 5], [6., float("nan")]]]).numpy()
    dim = 2
    keepdim = False
    dtype = None
    
    input_dict = {
        "input": input,
        "dim": dim,
        "keepdim": keepdim,
        "dtype": dtype
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nansum_2"] = nansum_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nansum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nansum_2'.")


check_valid('torch.nansum', generated_inputs['torch.nansum_2'], lib="torch", suffix=2)
