
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def multiheadattention_inputs():
    list_of_inputs = []

    input_dict = {
        "embed_dim": 512,
        "num_heads": 8,
        "dropout": 0.0,
        "bias": True,
        "add_bias_kv": False,
        "add_zero_attn": False,
        "kdim": None,
        "vdim": None,
        "batch_first": False,
        "dtype": np.float32,
        "query": np.random.rand(10, 512).astype(np.float32),
        "key": np.random.rand(10, 512).astype(np.float32),
        "value": np.random.rand(10, 512).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "embed_dim": 256,
        "num_heads": 4,
        "dropout": 0.1,
        "bias": False,
        "add_bias_kv": True,
        "add_zero_attn": True,
        "kdim": 128,
        "vdim": 64,
        "batch_first": True,
        "dtype": np.float32,
        "query": np.random.rand(5, 20, 256).astype(np.float32),
        "key": np.random.rand(5, 20, 128).astype(np.float32),
        "value": np.random.rand(5, 20, 128).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "embed_dim": 128,
        "num_heads": 2,
        "dropout": 0.2,
        "bias": True,
        "add_bias_kv": False,
        "add_zero_attn": False,
        "kdim": 64,
        "vdim": None,
        "batch_first": False,
        "dtype": np.float32,
        "query": np.random.rand(15, 128).astype(np.float32),
        "key": np.random.rand(15, 64).astype(np.float32),
        "value": np.random.rand(15, 64).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "embed_dim": 64,
        "num_heads": 1,
        "dropout": 0.3,
        "bias": False,
        "add_bias_kv": True,
        "add_zero_attn": True,
        "kdim": None,
        "vdim": 32,
        "batch_first": True,
        "dtype": np.float32,
        "query": np.random.rand(2, 30, 64).astype(np.float32),
        "key": np.random.rand(2, 30, 64).astype(np.float32),
        "value": np.random.rand(2, 30, 32).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "embed_dim": 1024,
        "num_heads": 16,
        "dropout": 0.05,
        "bias": True,
        "add_bias_kv": True,
        "add_zero_attn": False,
        "kdim": 512,
        "vdim": 256,
        "batch_first": False,
        "dtype": np.float32,
        "query": np.random.rand(20, 1024).astype(np.float32),
        "key": np.random.rand(20, 512).astype(np.float32),
        "value": np.random.rand(20, 256).astype(np.float32)
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
