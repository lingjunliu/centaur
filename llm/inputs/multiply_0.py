
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def multiply_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes (broadcasting)
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array([2.0], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    input1 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input2 = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D tensors
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: Complex tensors
    input1 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    input2 = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex64)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Scalar multiplication
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input2 = np.array(2.0, dtype=np.float32)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.multiply"] = multiply_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.multiply' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.multiply'.")

check_valid('torch.multiply', generated_inputs['torch.multiply'], lib="torch")
