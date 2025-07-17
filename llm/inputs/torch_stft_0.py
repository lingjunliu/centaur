
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def stft_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n_fft_val = 4
    hop_length_val = 2
    win_length_val = 4
    window_tensor = np.hanning(win_length_val)
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
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(10)
    n_fft_val = 8
    hop_length_val = 4
    win_length_val = 8
    window_tensor = np.hamming(win_length_val)
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
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.sin(np.linspace(0, 4 * np.pi, 200, endpoint=False))
    n_fft_val = 256
    hop_length_val = 64
    win_length_val = 256
    window_tensor = np.blackman(win_length_val)
    center_val = True
    pad_mode_val = "replicate"
    normalized_val = False
    onesided_val = True
    return_complex_val = True
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randn(2, 100)
    n_fft_val = 64
    hop_length_val = 16
    win_length_val = 64
    window_tensor = np.bartlett(win_length_val)
    center_val = False
    pad_mode_val = "reflect"
    normalized_val = True
    onesided_val = False
    return_complex_val = False
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    n_fft_val = 10
    hop_length_val = 5
    win_length_val = 10
    window_tensor = np.ones(win_length_val)
    center_val = True
    pad_mode_val = "constant"
    normalized_val = False
    onesided_val = True
    return_complex_val = True
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.arange(1, 21, dtype=np.float64)
    n_fft_val = 16
    hop_length_val = 8
    win_length_val = 16
    window_tensor = np.arange(1, win_length_val + 1, dtype=np.float64)
    center_val = False
    pad_mode_val = "reflect"
    normalized_val = True
    onesided_val = False
    return_complex_val = False
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    input_tensor = np.random.randn(3, 50)
    n_fft_val = 32
    hop_length_val = 8
    win_length_val = 32
    window_tensor = np.zeros(win_length_val)
    center_val = True
    pad_mode_val = "replicate"
    normalized_val = False
    onesided_val = True
    return_complex_val = True
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10], dtype=np.float32)
    n_fft_val = 10
    hop_length_val = 5
    win_length_val = 10
    window_tensor = np.ones(win_length_val) * -1
    center_val = False
    pad_mode_val = "circular"
    normalized_val = False
    onesided_val = False
    return_complex_val = False
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    n_fft_val = 16
    hop_length_val = 4
    win_length_val = 8
    window_tensor = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    center_val = True
    pad_mode_val = "constant"
    normalized_val = True
    onesided_val = True
    return_complex_val = True
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
        "center": center_val,
        "pad_mode": pad_mode_val,
        "normalized": normalized_val,
        "onesided": onesided_val,
        "return_complex": return_complex_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.randn(1, 256)
    n_fft_val = 512
    hop_length_val = 128
    win_length_val = 512
    window_tensor = np.random.rand(win_length_val)
    center_val = False
    pad_mode_val = "reflect"
    normalized_val = False
    onesided_val = False
    return_complex_val = False
    input_dict = {
        "input": input_tensor,
        "n_fft": n_fft_val,
        "hop_length": hop_length_val,
        "win_length": win_length_val,
        "window": window_tensor,
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
