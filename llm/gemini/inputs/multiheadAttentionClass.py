
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def multiheadattention_inputs():
    list_of_inputs = []

    # Input 1: Basic input
    embed_dim = 4
    num_heads = 2
    input_dict = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'dropout': 0.0,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': None,
        'vdim': None,
        'batch_first': False,
        'dtype': None,
        'query': np.random.rand(2, 3, embed_dim).astype(np.float32),
        'key': np.random.rand(2, 3, embed_dim).astype(np.float32),
        'value': np.random.rand(2, 3, embed_dim).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: batch_first=True
    embed_dim = 8
    num_heads = 4
    input_dict = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'dropout': 0.1,
        'bias': False,
        'add_bias_kv': True,
        'add_zero_attn': True,
        'kdim': None,
        'vdim': None,
        'batch_first': True,
        'dtype': None,
        'query': np.random.rand(3, 2, embed_dim).astype(np.float32),
        'key': np.random.rand(3, 2, embed_dim).astype(np.float32),
        'value': np.random.rand(3, 2, embed_dim).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: kdim and vdim specified
    embed_dim = 16
    num_heads = 8
    kdim = 8
    vdim = 8
    input_dict = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'dropout': 0.2,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': kdim,
        'vdim': vdim,
        'batch_first': False,
        'dtype': None,
        'query': np.random.rand(2, 3, embed_dim).astype(np.float32),
        'key': np.random.rand(2, 3, kdim).astype(np.float32),
        'value': np.random.rand(2, 3, vdim).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: dropout
    embed_dim = 32
    num_heads = 16
    input_dict = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'dropout': 0.3,
        'bias': False,
        'add_bias_kv': True,
        'add_zero_attn': True,
        'kdim': None,
        'vdim': None,
        'batch_first': True,
        'dtype': None,
        'query': np.random.rand(3, 2, embed_dim).astype(np.float32),
        'key': np.random.rand(3, 2, embed_dim).astype(np.float32),
        'value': np.random.rand(3, 2, embed_dim).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All default
    embed_dim = 64
    num_heads = 32
    input_dict = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'dropout': 0.0,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': None,
        'vdim': None,
        'batch_first': False,
        'dtype': None,
        'query': np.random.rand(2, 3, embed_dim).astype(np.float32),
        'key': np.random.rand(2, 3, embed_dim).astype(np.float32),
        'value': np.random.rand(2, 3, embed_dim).astype(np.float32)
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
