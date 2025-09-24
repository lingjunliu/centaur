
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cummin_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input1 = torch.randn(5).numpy()
    dim1 = 0
    out1 = (np.array([0.0], dtype=np.float32), np.array([0]))
    input_dict1 = {"input": input1, "dim": dim1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor
    input2 = torch.randn(2, 3).numpy()
    dim2 = 1
    out2 = (np.array([0.0], dtype=np.float32), np.array([0]))
    input_dict2 = {"input": input2, "dim": dim2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor
    input3 = torch.randn(2, 3, 4).numpy()
    dim3 = 0
    out3 = (np.array([0.0], dtype=np.float32), np.array([0]))
    input_dict3 = {"input": input3, "dim": dim3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values
    input4 = torch.tensor([-1.0, -2.0, 0.0, 1.0, 2.0]).numpy()
    dim4 = 0
    out4 = (np.array([0.0], dtype=np.float32), np.array([0]))
    input_dict4 = {"input": input4, "dim": dim4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor
    input5 = torch.randn(10, 10).numpy()
    dim5 = 0
    out5 = (np.array([0.0], dtype=np.float32), np.array([0]))
    input_dict5 = {"input": input5, "dim": dim5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cummin"] = cummin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cummin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cummin'.")

check_valid('torch.cummin', generated_inputs['torch.cummin'], lib="torch", suffix=0)
