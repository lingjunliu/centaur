
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def stft_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    n_fft_val = 4
    hop_length_val = 2
    win_length_val = 4
    window_val = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    center_val = True
    pad_mode_val = "reflect"
    normalized_val = False
    onesided_val = True
    return_complex_val = True
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_val,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    n_fft_val = 8
    hop_length_val = 4
    win_length_val = 8
    window_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    center_val = False
    pad_mode_val = "constant"
    normalized_val = True
    onesided_val = False
    return_complex_val = False
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_val,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.stft"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.stft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.stft'.")

check_valid('torch.stft', generated_inputs['torch.stft'], lib="torch", suffix=0)
