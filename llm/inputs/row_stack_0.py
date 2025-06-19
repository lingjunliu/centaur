
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def row_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    tensors1 = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
    input_dict1 = {"tensors": tensors1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensors
    tensors2 = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 6], [7, 8]], dtype=np.int32)]
    input_dict2 = {"tensors": tensors2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tensors with different shapes but compatible for row_stack
    tensors3 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    input_dict3 = {"tensors": tensors3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: 3D tensors
    tensors4 = [np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)]
    input_dict4 = {"tensors": tensors4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Single tensor in the list
    tensors5 = [np.array([[1, 2]])]
    input_dict5 = {"tensors": tensors5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Tensors with negative values and different data types.
    tensors6 = [np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64), np.array([[5, -6], [-7, 8]], dtype=np.int64)]
    input_dict6 = {"tensors": tensors6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Empty arrays
    tensors7 = [np.array([]), np.array([])]
    input_dict7 = {"tensors": tensors7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.row_stack"] = row_stack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.row_stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.row_stack'.")

check_valid('torch.row_stack', generated_inputs['torch.row_stack'], lib="torch")
