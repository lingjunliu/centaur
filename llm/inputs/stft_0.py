
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def stft_inputs():
    list_of_inputs = []

    # Input 1: Basic case with real input
    input1 = torch.randn(1000).numpy()
    n_fft1 = 512
    hop_length1 = 128
    window1 = torch.hann_window(n_fft1).numpy()

    input_dict1 = {
        "input": input1,
        "n_fft": n_fft1,
        "hop_length": hop_length1,
        "win_length": n_fft1,
        "window": window1,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Batch input with different parameters
    input2 = torch.randn(2, 1500).numpy()
    n_fft2 = 256
    hop_length2 = 64
    win_length2 = 200
    window2 = torch.hamming_window(win_length2).numpy()

    input_dict2 = {
        "input": input2,
        "n_fft": n_fft2,
        "hop_length": hop_length2,
        "win_length": win_length2,
        "window": window2,
        "center": False,
        "pad_mode": "constant",
        "normalized": True,
        "onesided": False,
        "return_complex": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: No window specified
    input3 = torch.randn(800).numpy()
    n_fft3 = 128

    input_dict3 = {
        "input": input3,
        "n_fft": n_fft3,
        "hop_length": 32,
        "win_length": None,
        "window": None,
        "center": True,
        "pad_mode": "replicate",
        "normalized": False,
        "onesided": True,
        "return_complex": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Complex input
    input4 = (torch.randn(500) + 1j * torch.randn(500)).numpy()
    n_fft4 = 64
    hop_length4 = 16
    window4 = torch.blackman_window(n_fft4).numpy()

    input_dict4 = {
        "input": input4,
        "n_fft": n_fft4,
        "hop_length": hop_length4,
        "win_length": n_fft4,
        "window": window4,
        "center": False,
        "pad_mode": "circular",
        "normalized": True,
        "onesided": False,
        "return_complex": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Input with win_length < n_fft
    input5 = torch.randn(1200).numpy()
    n_fft5 = 256
    win_length5 = 128
    hop_length5 = 32
    window5 = torch.bartlett_window(win_length5).numpy()

    input_dict5 = {
        "input": input5,
        "n_fft": n_fft5,
        "hop_length": hop_length5,
        "win_length": win_length5,
        "window": window5,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.stft"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.stft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.stft'.")

check_valid('torch.stft', generated_inputs['torch.stft'], lib="torch")
