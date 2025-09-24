
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def addcmul_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors, value=1
    input = np.random.randn(2, 3).astype(np.float32)
    tensor1 = np.random.randn(2, 3).astype(np.float32)
    tensor2 = np.random.randn(2, 3).astype(np.float32)
    value = 1.0
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Integer tensors, value=2
    input = np.random.randint(-5, 5, size=(3, 4)).astype(np.int32)
    tensor1 = np.random.randint(-5, 5, size=(3, 4)).astype(np.int32)
    tensor2 = np.random.randint(-5, 5, size=(3, 4)).astype(np.int32)
    value = 2
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Broadcasting, value=0.5
    input = np.random.randn(1, 5).astype(np.float64)
    tensor1 = np.random.randn(5, 1).astype(np.float64)
    tensor2 = np.random.randn(1, 5).astype(np.float64)
    value = 0.5
    out = np.empty((5,5), dtype=input.dtype)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Negative values and value=-1
    input = np.random.randn(4, 2).astype(np.float32)
    tensor1 = np.random.randn(4, 2).astype(np.float32)
    tensor2 = np.random.randn(4, 2).astype(np.float32)
    value = -1.0
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: 3D tensors
    input = np.random.randn(2, 3, 4).astype(np.float32)
    tensor1 = np.random.randn(2, 3, 4).astype(np.float32)
    tensor2 = np.random.randn(2, 3, 4).astype(np.float32)
    value = 0.25
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 6: Value as int
    input = np.random.randn(2, 3).astype(np.float32)
    tensor1 = np.random.randn(2, 3).astype(np.float32)
    tensor2 = np.random.randn(2, 3).astype(np.float32)
    value = 3
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 7: Scalar tensors
    input = np.random.randn(1).astype(np.float32)
    tensor1 = np.random.randn(1).astype(np.float32)
    tensor2 = np.random.randn(1).astype(np.float32)
    value = 1.5
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 8: Broadcasting with different shapes
    input = np.random.randn(5, 3).astype(np.float32)
    tensor1 = np.random.randn(5, 1).astype(np.float32)
    tensor2 = np.random.randn(1, 3).astype(np.float32)
    value = 0.75
    out = np.empty_like(input)
    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addcmul"] = addcmul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addcmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addcmul'.")

check_valid('torch.addcmul', generated_inputs['torch.addcmul'], lib="torch")
