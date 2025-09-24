
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_stft_inputs():
    list_of_inputs = []

    # Input 1: Basic example with real input
    input1 = torch.randn(1000).numpy()
    n_fft1 = 256
    hop_length1 = 64
    win_length1 = 256
    window1 = torch.hann_window(win_length1).numpy()
    return_complex1 = True

    input_dict1 = {
        "input": input1,
        "n_fft": n_fft1,
        "hop_length": hop_length1,
        "win_length": win_length1,
        "window": window1,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex1,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Complex input
    input2 = torch.randn(512, dtype=torch.complex64).numpy()
    n_fft2 = 128
    hop_length2 = 32
    win_length2 = 128
    window2 = torch.hamming_window(win_length2).numpy()
    return_complex2 = True

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
        "return_complex": return_complex2,
        "align_to_window": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Batch input (2D)
    input3 = torch.randn(2, 768).numpy()
    n_fft3 = 128
    hop_length3 = 64
    win_length3 = 128
    window3 = torch.bartlett_window(win_length3).numpy()
    return_complex3 = True

    input_dict3 = {
        "input": input3,
        "n_fft": n_fft3,
        "hop_length": hop_length3,
        "win_length": win_length3,
        "window": window3,
        "center": True,
        "pad_mode": "replicate",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex3,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: No window
    input4 = torch.randn(2048).numpy()
    n_fft4 = 512
    hop_length4 = 128
    win_length4 = 512
    return_complex4 = True
    input_dict4 = {
        "input": input4,
        "n_fft": n_fft4,
        "hop_length": hop_length4,
        "win_length": win_length4,
        "window": None,
        "center": False,
        "pad_mode": "circular",
        "normalized": True,
        "onesided": False,
        "return_complex": return_complex4,
        "align_to_window": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different parameters
    input5 = torch.randn(1, 500).numpy()
    n_fft5 = 256
    hop_length5 = 128
    win_length5 = 128
    window5 = torch.blackman_window(win_length5).numpy()
    return_complex5 = True
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
        "return_complex": return_complex5,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Negative values in input
    input6 = (torch.rand(500) - 0.5).numpy()
    n_fft6 = 128
    hop_length6 = 64
    win_length6 = 128
    window6 = torch.hann_window(win_length6).numpy()
    return_complex6 = True

    input_dict6 = {
        "input": input6,
        "n_fft": n_fft6,
        "hop_length": hop_length6,
        "win_length": win_length6,
        "window": window6,
        "center": True,
        "pad_mode": "reflect",
        "normalized": False,
        "onesided": True,
        "return_complex": return_complex6,
        "align_to_window": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = torch_stft_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('stft', generated_inputs)
