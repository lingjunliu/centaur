
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fake_quantize_per_channel_affine_inputs():
    list_of_inputs = []

    def _create_input(input_shape, ch_axis, quant_min, quant_max, in_dtype=np.float32, q_dtype=np.float32, zp_dtype=np.int32):
        """Helper to create a valid input dictionary."""
        num_channels = input_shape[ch_axis]
        scale = (np.random.rand(num_channels) * 2 + 0.1).astype(q_dtype)
        
        if quant_min == quant_max:
            zero_point_val = np.full(num_channels, quant_min)
        else:
            zero_point_val = np.random.randint(quant_min, quant_max + 1, size=num_channels)
            
        zero_point = zero_point_val.astype(zp_dtype)
        
        return {
            'input': (np.random.randn(*input_shape) * 10).astype(in_dtype),
            'scale': scale,
            'zero_point': zero_point,
            'quant_min': quant_min,
            'quant_max': quant_max,
            'ch_axis': ch_axis
        }

    # Input 1: 4D NCHW, 8-bit unsigned
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(2, 4, 5, 5), ch_axis=1, quant_min=0, quant_max=255)))

    # Input 2: 4D NHWC, 8-bit signed
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(1, 8, 8, 3), ch_axis=3, quant_min=-128, quant_max=127)))

    # Input 3: 2D (FC weights), ch_axis=0, 8-bit signed asymmetric
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(6, 10), ch_axis=0, quant_min=-127, quant_max=127)))

    # Input 4: 2D (FC weights), ch_axis=1, 4-bit unsigned
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(8, 4), ch_axis=1, quant_min=0, quant_max=15)))

    # Input 5: 3D tensor, ch_axis=1, 8-bit signed
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(3, 5, 7), ch_axis=1, quant_min=-128, quant_max=127)))
    
    # Input 6: 5D tensor, ch_axis=1, 7-bit signed
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(2, 3, 2, 2, 2), ch_axis=1, quant_min=-64, quant_max=63)))

    # Input 7: quant_min == quant_max (edge case)
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(4, 2), ch_axis=0, quant_min=5, quant_max=5)))

    # Input 8: 4D NCHW, ch_axis=0 (batch dim quantization)
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(4, 3, 6, 6), ch_axis=0, quant_min=-128, quant_max=127)))

    # Input 9: float64 input type with int64 zero_point
    list_of_inputs.append(copy.deepcopy(_create_input(
        input_shape=(3, 2), ch_axis=1, quant_min=-128, quant_max=127,
        in_dtype=np.float64, q_dtype=np.float64, zp_dtype=np.int64
    )))
    
    # Input 10: 1D tensor
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(20,), ch_axis=0, quant_min=0, quant_max=255)))

    # Input 11: Another 2D case with negative channel axis
    list_of_inputs.append(copy.deepcopy(_create_input(input_shape=(10, 5), ch_axis=-1, quant_min=0, quant_max=255)))

    return list_of_inputs

generated_inputs["torch.fake_quantize_per_channel_affine"] = fake_quantize_per_channel_affine_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fake_quantize_per_channel_affine' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fake_quantize_per_channel_affine'.")

check_valid('torch.fake_quantize_per_channel_affine', generated_inputs['torch.fake_quantize_per_channel_affine'], lib="torch", suffix=0)
