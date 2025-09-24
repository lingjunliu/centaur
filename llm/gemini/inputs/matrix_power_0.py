
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_power_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix with positive power
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    n1 = 2
    input_dict1 = {"input": input1, "n": n1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Square matrix with negative power (invertible)
    input2 = np.array([[1, 1], [1, 2]], dtype=np.float64)
    n2 = -1
    input_dict2 = {"input": input2, "n": n2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Square matrix with zero power
    input3 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    n3 = 0
    input_dict3 = {"input": input3, "n": n3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger square matrix with positive power
    input4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    n4 = 3
    input_dict4 = {"input": input4, "n": n4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Square matrix with decimal values and larger power
    input5 = np.array([[0.5, 0.2], [0.1, 0.9]], dtype=np.float32)
    n5 = 5
    input_dict5 = {"input": input5, "n": n5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Complex square matrix with positive power
    input6 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    n6 = 2
    input_dict6 = {"input": input6, "n": n6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Another matrix with negative power (invertible)
    input7 = np.array([[2, 1], [1, 1]], dtype=np.float64)
    n7 = -2
    input_dict7 = {"input": input7, "n": n7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.matrix_power"] = matrix_power_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.matrix_power' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.matrix_power'.")

check_valid('torch.matrix_power', generated_inputs['torch.matrix_power'], lib="torch")
