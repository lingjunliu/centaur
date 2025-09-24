
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def stft_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1000).numpy()
    n_fft = 256
    hop_length = 128
    win_length = 256
    window = torch.hann_window(win_length).numpy()
    center = True
    pad_mode = "reflect"
    normalized = False
    onesided = True
    return_complex = True

    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "center": center,
        "pad_mode": pad_mode,
        "normalized": normalized,
        "onesided": onesided,
        "return_complex": return_complex,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(500).numpy()
    n_fft = 128
    hop_length = 64
    win_length = 128
    window = torch.hamming_window(win_length).numpy()
    center = False
    pad_mode = "constant"
    normalized = True
    onesided = False
    return_complex = False

    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "center": center,
        "pad_mode": pad_mode,
        "normalized": normalized,
        "onesided": onesided,
        "return_complex": return_complex,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.stft"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.stft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.stft'.")

check_valid('torch.stft', generated_inputs['torch.stft'], lib="torch", suffix=0)
