
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def multi_dot_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    c = np.random.rand(4, 2)
    tensors = [a, b, c]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    a = np.random.randint(1, 5, (2, 3))
    b = np.random.randint(1, 5, (3, 4))
    c = np.random.randint(1, 5, (4, 2))
    tensors = [a, b, c]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different dimensions
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    c = np.random.rand(4, 5)
    d = np.random.rand(5, 2)
    tensors = [a, b, c, d]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Only two matrices
    a = np.random.rand(5, 3)
    b = np.random.rand(3, 6)
    tensors = [a, b]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: More matrices, different shapes
    a = np.random.rand(1, 2)
    b = np.random.rand(2, 3)
    c = np.random.rand(3, 4)
    d = np.random.rand(4, 1)
    e = np.random.rand(1, 5)
    f = np.random.rand(5, 1)
    tensors = [a, b, c, d, e, f]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Negative Values
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)
    c = np.random.randn(4, 2)
    tensors = [a, b, c]
    input_dict = {"tensors": tensors}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.multi_dot"] = multi_dot_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.multi_dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.multi_dot'.")

check_valid('torch.linalg.multi_dot', generated_inputs['torch.linalg.multi_dot'], lib="torch")
