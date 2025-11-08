
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dropout2d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(20, 16, 32, 32, dtype=torch.float32).numpy()
    p = np.float32(0.2)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 2
    input_arr = torch.randn(1, 1, 8, 8, dtype=torch.float64).numpy()
    p = np.float64(0.0)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 3
    input_arr = torch.randn(3, 3, 10, 20, dtype=torch.float32).numpy()
    p = np.float32(1.0)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 4
    input_arr = torch.randn(5, 2, 1, 1, dtype=torch.float32).numpy()
    p = np.float32(0.5)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 5 (3D NCL)
    input_arr = torch.randn(2, 4, 15, dtype=torch.float32).numpy()
    p = np.float32(0.8)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 6
    input_arr = torch.randn(4, 7, 9, 5, dtype=torch.float32).numpy()
    p = np.float32(0.33)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 7 (3D NCL)
    input_arr = torch.randn(10, 5, 7, dtype=torch.float32).numpy()
    p = np.float32(0.05)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 8
    input_arr = torch.randn(2, 32, 64, 64, dtype=torch.float32).numpy()
    p = np.float32(0.75)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 9
    input_arr = torch.randn(8, 1, 3, 5, dtype=torch.float32).numpy()
    p = np.float32(0.9)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 10 (float16)
    input_arr = torch.randn(6, 6, 12, 12, dtype=torch.float16).numpy()
    p = np.float16(0.1)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 11
    input_arr = torch.randn(2, 8, 4, 6, dtype=torch.float32).numpy()
    p = np.float32(0.66)
    inplace = np.bool_(True)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    # 12 (3D NCL)
    input_arr = torch.randn(1, 16, 128, dtype=torch.float32).numpy()
    p = np.float32(0.25)
    inplace = np.bool_(False)
    list_of_inputs.append(copy.deepcopy({"p": p, "inplace": inplace, "input": input_arr}))

    return list_of_inputs

generated_inputs["torch.nn.Dropout2d"] = dropout2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Dropout2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Dropout2d'.")


check_valid('torch.nn.Dropout2d', generated_inputs['torch.nn.Dropout2d'], lib="torch", suffix=0)
