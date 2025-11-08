
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def signbit_inputs():
    list_of_inputs = []
    
    # 1
    input_arr = np.array([0.7, -1.2, 0.0, 2.3], dtype=np.float32)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 2
    input_arr = np.array([-0.0, 0.0, -np.inf, np.inf, np.nan, -np.nan], dtype=np.float64)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 3
    input_arr = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, -0.0]], dtype=np.float16)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 4
    input_arr = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 5
    input_arr = np.array(-0.0, dtype=np.float64)
    out_arr = np.empty((), dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 6
    input_arr = np.array([[[-1, 0], [1, -2]], [[3, -3], [4, 5]]], dtype=np.int8)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 7
    input_arr = np.arange(0, 12, dtype=np.uint8).reshape(3, 4)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 8
    base = np.linspace(-5, 5, 12, dtype=np.float32).reshape(3, 4)
    input_arr = base[:, ::-1]
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 9
    input_arr = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 2, 2)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 10
    input_arr = np.array([-(2**63) + 1, -123, 0, 123, (2**63) - 1], dtype=np.int64)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 11
    input_arr = np.array([-0.0, 0.0, -0.0, 1.0, -1.0], dtype=np.float32)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    # 12
    input_arr = np.array([-1e-8, 1e-8, -5e-5, 5e-5], dtype=np.float16)
    out_arr = np.empty(input_arr.shape, dtype=bool)
    list_of_inputs.append(copy.deepcopy({"input": input_arr, "out": out_arr}))
    
    return list_of_inputs

generated_inputs["torch.signbit"] = signbit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.signbit'.")


check_valid('torch.signbit', generated_inputs['torch.signbit'], lib="torch", suffix=0)
