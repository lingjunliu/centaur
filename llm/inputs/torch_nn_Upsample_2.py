
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def upsample_inputs():
    list_of_inputs = []

    # Case 1: 2D input, using scale_factor
    input_dict_1 = {
        'input': np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        'scale_factor': 2.0,
        'mode': 'bilinear',
        'align_corners': False,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_1)

    # Case 2: 2D input, using size
    input_dict_2 = {
        'input': np.random.rand(2, 3, 4, 5).astype(np.float32),
        'size': (8, 10),
        'mode': 'nearest',
        'align_corners': False,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_2)

    # Case 3: 2D input, bicubic interpolation with align_corners=True
    input_dict_3 = {
        'input': np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3),
        'scale_factor': 2.0,
        'mode': 'bicubic',
        'align_corners': True,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_3)
    
    # Case 4: 2D input, with recompute_scale_factor=True
    input_dict_4 = {
        'input': np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        'scale_factor': 2.5,
        'mode': 'bilinear',
        'align_corners': False,
        'recompute_scale_factor': True,
    }
    list_of_inputs.append(input_dict_4)

    # Case 5: 1D input, using scale_factor
    input_dict_5 = {
        'input': np.random.rand(2, 2, 10).astype(np.float32),
        'scale_factor': 2.0,
        'mode': 'linear',
        'align_corners': False,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_5)

    # Case 6: 1D input, using size
    input_dict_6 = {
        'input': np.arange(1, 9, dtype=np.float32).reshape(1, 1, 8),
        'size': (20,),
        'mode': 'linear',
        'align_corners': True,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_6)

    # Case 7: 3D input, using scale_factor
    input_dict_7 = {
        'input': np.arange(1, 9, dtype=np.float32).reshape(1, 1, 2, 2, 2),
        'scale_factor': 2.0,
        'mode': 'trilinear',
        'align_corners': False,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_7)

    # Case 8: 3D input, using size
    input_dict_8 = {
        'input': np.random.rand(1, 2, 3, 4, 5).astype(np.float32),
        'size': (6, 8, 10),
        'mode': 'trilinear',
        'align_corners': True,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_8)

    # Case 9: 3D input, nearest neighbor with scale_factor
    input_dict_9 = {
        'input': np.arange(1, 28, dtype=np.float32).reshape(1, 1, 3, 3, 3),
        'scale_factor': 1.5,
        'mode': 'nearest',
        'align_corners': False,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_9)

    # Case 10: large input tensor with size
    input_dict_10 = {
        'input': np.random.rand(4, 8, 16, 16).astype(np.float32),
        'size': (32, 32),
        'mode': 'bilinear',
        'align_corners': False,
        'recompute_scale_factor': False,
    }
    list_of_inputs.append(input_dict_10)

    # The test harness requires all keys from the signature. We add the missing
    # one with a value that should be ignored by PyTorch, but satisfies the harness.
    # This is a workaround for the conflicting requirements.
    final_list = []
    for d in list_of_inputs:
        d_copy = copy.deepcopy(d)
        if 'size' not in d_copy:
            d_copy['size'] = (0,) * (d_copy['input'].ndim - 2)
        if 'scale_factor' not in d_copy:
            # Using 0.0 would cause a ValueError. Using 1.0 is a "no-op" scale factor
            # that will be ignored by PyTorch since 'size' is provided.
            d_copy['scale_factor'] = 1.0
        final_list.append(d_copy)

    # Since the above strategy can still lead to a `ValueError`,
    # the following strategy of providing `None` is semantically correct for PyTorch,
    # even if the test harness cannot handle it.
    final_list_with_none = []
    for d in list_of_inputs:
        d_copy = copy.deepcopy(d)
        if 'size' not in d_copy:
            d_copy['size'] = None
        if 'scale_factor' not in d_copy:
            d_copy['scale_factor'] = None
        final_list_with_none.append(d_copy)


    return final_list_with_none

generated_inputs["torch.nn.Upsample_2"] = upsample_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Upsample_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Upsample_2'.")

check_valid('torch.nn.Upsample', generated_inputs['torch.nn.Upsample_2'], lib="torch", suffix=2)
