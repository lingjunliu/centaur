
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def multiheadattention_inputs():
    list_of_inputs = []

    input_dict = {
        'embed_dim': 512,
        'num_heads': 8,
        'dropout': 0.1,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': 512,
        'vdim': 512,
        'batch_first': False,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'embed_dim': 256,
        'num_heads': 4,
        'dropout': 0.0,
        'bias': False,
        'add_bias_kv': True,
        'add_zero_attn': True,
        'kdim': None,
        'vdim': None,
        'batch_first': True,
        'dtype': torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'embed_dim': 1024,
        'num_heads': 16,
        'dropout': 0.2,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': None,
        'vdim': None,
        'batch_first': False,
        'dtype': torch.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'embed_dim': 64,
        'num_heads': 2,
        'dropout': 0.5,
        'bias': False,
        'add_bias_kv': True,
        'add_zero_attn': True,
        'kdim': None,
        'vdim': None,
        'batch_first': True,
        'dtype': torch.bfloat16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        'embed_dim': 128,
        'num_heads': 1,
        'dropout': 0.0,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': None,
        'vdim': None,
        'batch_first': False,
        'dtype': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = multiheadattention_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multiheadAttentionClass', generated_inputs)
