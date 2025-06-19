
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_polar_inputs():
    list_of_inputs = []

    # Case 1: float64, simple 1D tensors
    abs_val = np.array([1.0, 2.0], dtype=np.float64)
    angle_val = np.array([np.pi/2, np.pi], dtype=np.float64)
    input_dict = {"abs": abs_val, "angle": angle_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: float32, simple 1D tensors
    abs_val = np.array([1.0, 2.0], dtype=np.float32)
    angle_val = np.array([np.pi/2, np.pi], dtype=np.float32)
    input_dict = {"abs": abs_val, "angle": angle_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: float64, 2D tensors
    abs_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    angle_val = np.array([[np.pi/2, np.pi], [0, np.pi/4]], dtype=np.float64)
    input_dict = {"abs": abs_val, "angle": angle_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: float32, 2D tensors
    abs_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    angle_val = np.array([[np.pi/2, np.pi], [0, np.pi/4]], dtype=np.float32)
    input_dict = {"abs": abs_val, "angle": angle_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: float64, different values including negative abs which is invalid but should be tested
    abs_val = np.array([-1.0, 2.0, 0.0], dtype=np.float64)
    angle_val = np.array([np.pi/2, np.pi, np.pi/4], dtype=np.float64)
    input_dict = {"abs": abs_val, "angle": angle_val, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.polar"] = torch_polar_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.polar' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.polar'.")

check_valid('torch.polar', generated_inputs['torch.polar'], lib="torch")
