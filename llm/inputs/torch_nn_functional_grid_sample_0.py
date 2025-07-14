
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def grid_sample_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
    grid_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32) * 2 - 1
    mode_str = "bilinear"
    padding_mode_str = "zeros"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 4, 20, 20).astype(np.float32)
    grid_tensor = np.random.rand(2, 10, 10, 2).astype(np.float32) * 2 - 1
    mode_str = "nearest"
    padding_mode_str = "border"
    align_corners_bool = False
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 1, 5, 5).astype(np.float32)
    grid_tensor = np.random.rand(1, 3, 3, 2).astype(np.float32) * 2 - 1
    mode_str = "bilinear"
    padding_mode_str = "reflection"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D input and grid
    input_tensor = np.random.rand(1, 3, 5, 5, 5).astype(np.float32)
    grid_tensor = np.random.rand(1, 3, 3, 3, 3).astype(np.float32) * 2 - 1
    mode_str = "bilinear"
    padding_mode_str = "zeros"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: different batch size
    input_tensor = np.random.rand(4, 3, 10, 10).astype(np.float32)
    grid_tensor = np.random.rand(4, 5, 5, 2).astype(np.float32) * 2 - 1
    mode_str = "bilinear"
    padding_mode_str = "zeros"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small input size
    input_tensor = np.random.rand(1, 3, 2, 2).astype(np.float32)
    grid_tensor = np.random.rand(1, 1, 1, 2).astype(np.float32) * 2 - 1
    mode_str = "bilinear"
    padding_mode_str = "zeros"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Nearest mode with align_corners False
    input_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
    grid_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32) * 2 - 1
    mode_str = "nearest"
    padding_mode_str = "zeros"
    align_corners_bool = False
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Border padding mode
    input_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
    grid_tensor = np.random.rand(1, 5, 5, 2).astype(np.float32) * 2 - 1
    mode_str = "bilinear"
    padding_mode_str = "border"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 1, 3, 3).astype(np.float32)
    grid_tensor = np.array([[[[-1, -1], [-1, 0], [-1, 1]],
                              [[0, -1], [0, 0], [0, 1]],
                              [[1, -1], [1, 0], [1, 1]]]]).astype(np.float32)
    mode_str = "bilinear"
    padding_mode_str = "zeros"
    align_corners_bool = False
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.random.rand(2, 2, 4, 5).astype(np.float32)
    grid_tensor = np.random.rand(2, 3, 4, 2).astype(np.float32) * 2 - 1
    mode_str = "nearest"
    padding_mode_str = "reflection"
    align_corners_bool = True
    input_dict = {"input": input_tensor, "grid": grid_tensor, "mode": mode_str, "padding_mode": padding_mode_str, "align_corners": align_corners_bool}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.grid_sample"] = grid_sample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.grid_sample' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.grid_sample'.")

check_valid('torch.nn.functional.grid_sample', generated_inputs['torch.nn.functional.grid_sample'], lib="torch", suffix=0)
