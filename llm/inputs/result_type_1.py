
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def result_type_inputs():
    list_of_inputs = []

    tensor1 = np.array([1, 2, 3], dtype=np.float32)
    tensor2 = np.array([4, 5, 6], dtype=np.int64)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor1 = np.array([1, 2, 3], dtype=np.int8)
    tensor2 = np.array([4, 5, 6], dtype=np.uint8)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor1 = np.array([1, 2, 3], dtype=np.float64)
    tensor2 = np.array([4, 5, 6], dtype=np.float16)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor1 = np.array([1, 2, 3], dtype=np.complex64)
    tensor2 = np.array([4, 5, 6], dtype=np.float32)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor1 = np.array([1, 2, 3], dtype=np.int32)
    tensor2 = np.array([4, 5, 6], dtype=np.complex128)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    tensor1 = np.array([[1, 2], [3, 4]], dtype=np.int16)
    tensor2 = np.array([4, 5, 6, 7], dtype=np.int8)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    tensor1 = np.array([1], dtype=np.bool_)
    tensor2 = np.array([0], dtype=np.int64)
    input_dict = {
        "tensor1": tensor1,
        "tensor2": tensor2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.result_type_1"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.result_type_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.result_type_1'.")

check_valid('torch.result_type', generated_inputs['torch.result_type_1'], lib="torch")
