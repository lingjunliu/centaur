
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def true_divide_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[0.5, 1.0], [1.5, 2.0]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    input1 = np.array([[1, 2], [3, 4]])
    input2 = np.array([[2, 2], [2, 2]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Scalar division
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array(2.0)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    input1 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    input2 = np.array([[0.5, -1.0], [-1.5, 2.0]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Different shapes (broadcasting)
    input1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input2 = np.array([1.0, 2.0, 3.0])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D tensors
    input1 = np.random.rand(2, 3, 4)
    input2 = np.random.rand(2, 3, 4)
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Zero division (should not error, returns inf)
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 8: Complex numbers
    input1 = np.array([[1 + 1j, 2 - 2j], [3 + 0j, 4 - 1j]])
    input2 = np.array([[0.5, 1], [1.5, 2]])
    input_dict = {"input": input1, "other": input2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.true_divide"] = true_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.true_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.true_divide'.")

check_valid('torch.true_divide', generated_inputs['torch.true_divide'], lib="torch")
