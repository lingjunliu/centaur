
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_less_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = torch.randn(2, 3).numpy()
    other1 = torch.randn(2, 3).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Integer tensors
    input2 = torch.randint(-5, 5, (3, 4)).numpy()
    other2 = torch.randint(-5, 5, (3, 4)).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: Different shapes (broadcasting)
    input3 = torch.randn(2, 3, 4).numpy()
    other3 = torch.randn(4).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Case 4: Scalar
    input4 = torch.randn(5).numpy()
    other4 = 2.0
    input_dict4 = {"input": input4, "other": np.array(other4), "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: Boolean tensors
    input5 = torch.randn(2, 2).numpy() > 0.5
    other5 = torch.randn(2, 2).numpy() > 0.2
    input_dict5 = {"input": input5, "other": other5, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Case 6: 0-dimensional tensors
    input6 = torch.randn(()).numpy()
    other6 = torch.randn(()).numpy()
    input_dict6 = {"input": input6, "other": other6, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: Negative values
    input7 = torch.randn(2, 2) * -1.0
    other7 = torch.randn(2, 2) * -1.0
    input_dict7 = {"input": input7.numpy(), "other": other7.numpy(), "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs["torch.less"] = torch_less_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.less' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.less'.")

check_valid('torch.less', generated_inputs['torch.less'], lib="torch")
