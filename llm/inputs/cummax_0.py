
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cummax_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor
    input1 = torch.tensor([-0.3449, -1.5447, 0.0685, -1.5104, -1.1706, 0.2259, 1.4696, -1.3284, 1.9946, -0.8209]).numpy()
    dim1 = 0
    out1 = (np.zeros_like(input1), np.zeros_like(input1, dtype=np.int64))
    input_dict1 = {"input": input1, "dim": dim1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor, dim=0
    input2 = torch.randn(3, 4).numpy()
    dim2 = 0
    out2 = (np.zeros_like(input2), np.zeros_like(input2, dtype=np.int64))
    input_dict2 = {"input": input2, "dim": dim2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor, dim=1
    input3 = torch.randn(3, 4).numpy()
    dim3 = 1
    out3 = (np.zeros_like(input3), np.zeros_like(input3, dtype=np.int64))
    input_dict3 = {"input": input3, "dim": dim3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D tensor, dim=0
    input4 = torch.randn(2, 3, 4).numpy()
    dim4 = 0
    out4 = (np.zeros_like(input4), np.zeros_like(input4, dtype=np.int64))
    input_dict4 = {"input": input4, "dim": dim4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor, dim=1
    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = 1
    out5 = (np.zeros_like(input5), np.zeros_like(input5, dtype=np.int64))
    input_dict5 = {"input": input5, "dim": dim5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.cummax"] = cummax_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cummax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cummax'.")

check_valid('torch.cummax', generated_inputs['torch.cummax'], lib="torch", suffix=0)
