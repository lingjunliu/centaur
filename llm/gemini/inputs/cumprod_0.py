
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def cumprod_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor of integers
    input1 = np.array([1, 2, 3, 4, 5])
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor of floats
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with negative values
    input3 = np.array([[[1, -2], [3, 4]], [[5, 6], [-7, 8]]])
    dim3 = 2
    input_dict3 = {"input": input3, "dim": dim3, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor with dtype specified
    input4 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    dim4 = 0
    dtype4 = torch.float64
    input_dict4 = {"input": input4, "dim": dim4, "dtype": dtype4, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D tensor with out specified
    input5 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    dim5 = 0
    out5 = torch.empty_like(torch.from_numpy(input5))
    input_dict5 = {"input": input5, "dim": dim5, "dtype": None, "out": out5.numpy()}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Empty tensor
    input6 = np.array([])
    dim6 = 0
    input_dict6 = {"input": input6, "dim": dim6, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 1D tensor of booleans
    input7 = np.array([True, False, True])
    dim7 = 0
    input_dict7 = {"input": input7, "dim": dim7, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: 2D tensor of complex numbers
    input8 = np.array([[1+1j, 2-2j], [3+0j, 4+4j]])
    dim8 = 1
    input_dict8 = {"input": input8, "dim": dim8, "dtype": None, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.cumprod"] = cumprod_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cumprod'.")

check_valid('torch.cumprod', generated_inputs['torch.cumprod'], lib="torch")
