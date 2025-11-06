
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, numpy as np, copy

def ones_like_inputs():
    list_of_inputs = []
    
    input_arr = np.array([0.0, -1.5, 3.2], dtype=np.float32)
    dtype = torch.float32
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.array([[1, -2, 3], [0, 5, -6]], dtype=np.int64)
    dtype = torch.int64
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.array([True, False, True, True], dtype=np.bool_)
    dtype = torch.bool
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.array(2.718281828, dtype=np.float64)
    dtype = torch.float64
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.zeros((2, 0, 3, 4), dtype=np.float16)
    dtype = torch.float16
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.array([[1+2j, -3-4j], [0+0j, 5-6j]], dtype=np.complex64)
    dtype = torch.complex64
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.array([255, 0, 128], dtype=np.uint8)
    dtype = torch.uint8
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    dtype = torch.int32
    requires_grad = False
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.random.randn(3, 3).astype(np.float32)
    dtype = torch.float32
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.arange(12, dtype=np.float32).reshape(3, 4).transpose(1, 0)
    dtype = torch.float32
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.arange(10, dtype=np.float64)[::-1].copy()
    dtype = torch.float64
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    input_arr = np.array([[-1, 2], [3, -4]], dtype=np.int16)
    dtype = torch.float32
    requires_grad = True
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "dtype": dtype, "requires_grad": requires_grad}))
    
    return list_of_inputs

generated_inputs["torch.ones_like"] = ones_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ones_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_like'.")


check_valid('torch.ones_like', generated_inputs['torch.ones_like'], lib="torch", suffix=0)
