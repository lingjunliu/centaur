
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy

def stft_inputs():
    list_of_inputs = []
    
    # Helper values
    L = 1024
    n_fft_val = 400
    win_length_val = 400
    hop_length_val = 100

    # Input 1: Basic 1D real input, common case
    input_dict1 = {
        'input': torch.randn(L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D (batched) real input
    input_dict2 = {
        'input': torch.randn(3, L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: center=False, requires input length >= n_fft
    input_dict3 = {
        'input': torch.randn(L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': False,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: onesided=False to get full spectrum
    input_dict4 = {
        'input': torch.randn(L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': False,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: return_complex=False (deprecated but valid)
    input_dict5 = {
        'input': torch.randn(L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': True,
        'return_complex': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different pad_mode, normalized=True
    input_dict6 = {
        'input': torch.randn(L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'constant',
        'normalized': True,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: win_length < n_fft
    win_length_small = 200
    input_dict7 = {
        'input': torch.randn(L).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_small,
        'window': torch.hann_window(win_length_small).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 1D Complex input (requires onesided=False)
    input_dict8 = {
        'input': torch.randn(L, dtype=torch.cfloat).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': False,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Short input signal (L < n_fft) with another pad_mode
    L_short = 200
    input_dict9 = {
        'input': torch.randn(L_short).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': True,
        'pad_mode': 'replicate',
        'normalized': False,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Batched Complex input with return_complex=False
    input_dict10 = {
        'input': torch.randn(2, L, dtype=torch.cfloat).numpy(),
        'n_fft': n_fft_val,
        'hop_length': hop_length_val,
        'win_length': win_length_val,
        'window': torch.hann_window(win_length_val).numpy(),
        'center': False,
        'pad_mode': 'reflect',
        'normalized': True,
        'onesided': False,
        'return_complex': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Different window (Hamming) and different fft parameters
    input_dict11 = {
        'input': torch.randn(L).numpy(),
        'n_fft': 512,
        'hop_length': 128,
        'win_length': 512,
        'window': torch.hamming_window(512).numpy(),
        'center': True,
        'pad_mode': 'reflect',
        'normalized': False,
        'onesided': True,
        'return_complex': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))
    
    return list_of_inputs

generated_inputs["torch.stft"] = stft_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.stft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.stft'.")

check_valid('torch.stft', generated_inputs['torch.stft'], lib="torch", suffix=0)
