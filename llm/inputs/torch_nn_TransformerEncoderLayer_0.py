
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def transformer_encoder_layer_inputs():
    list_of_inputs = []

    # Input 1
    src = np.random.rand(10, 32, 512).astype(np.float32)
    src_mask = np.random.rand(10, 10).astype(np.float32)
    src_key_padding_mask = np.random.randint(0, 2, (32, 10)).astype(bool)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    src = np.random.rand(32, 10, 512).astype(np.float32)
    src_mask = None
    src_key_padding_mask = None
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.2,
        "activation": "gelu",
        "layer_norm_eps": 1e-06,
        "batch_first": True,
        "norm_first": True,
        "bias": False,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    src = np.random.rand(5, 16, 256).astype(np.float64)
    src_mask = np.triu(np.ones((5, 5))).astype(bool)
    src_key_padding_mask = np.zeros((16, 5)).astype(bool)
    input_dict = {
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 512,
        "dropout": 0.05,
        "activation": "relu",
        "layer_norm_eps": 1e-07,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": torch.float64,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    src = np.random.rand(16, 5, 256).astype(np.float64)
    src_mask = None
    src_key_padding_mask = None
    input_dict = {
        "d_model": 256,
        "nhead": 4,
        "dim_feedforward": 512,
        "dropout": 0.05,
        "activation": "gelu",
        "layer_norm_eps": 1e-07,
        "batch_first": True,
        "norm_first": True,
        "bias": False,
        "dtype": torch.float64,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    src = np.random.rand(20, 64, 128).astype(np.float32)
    src_mask = np.zeros((20, 20)).astype(bool)
    src_key_padding_mask = np.ones((64, 20)).astype(bool)
    input_dict = {
        "d_model": 128,
        "nhead": 2,
        "dim_feedforward": 256,
        "dropout": 0.3,
        "activation": "relu",
        "layer_norm_eps": 1e-08,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    src = np.random.rand(64, 20, 128).astype(np.float32)
    src_mask = None
    src_key_padding_mask = None
    input_dict = {
        "d_model": 128,
        "nhead": 2,
        "dim_feedforward": 256,
        "dropout": 0.3,
        "activation": "gelu",
        "layer_norm_eps": 1e-08,
        "batch_first": True,
        "norm_first": True,
        "bias": False,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    src = np.random.rand(4, 8, 32).astype(np.float32)
    src_mask = np.random.rand(4, 4).astype(np.float32)
    src_key_padding_mask = np.random.randint(0, 2, (8, 4)).astype(bool)
    input_dict = {
        "d_model": 32,
        "nhead": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "activation": "relu",
        "layer_norm_eps": 1e-12,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    src = np.random.rand(8, 4, 32).astype(np.float32)
    src_mask = None
    src_key_padding_mask = None
    input_dict = {
        "d_model": 32,
        "nhead": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "activation": "gelu",
        "layer_norm_eps": 1e-12,
        "batch_first": True,
        "norm_first": True,
        "bias": False,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    src = np.random.rand(10, 32, 512).astype(np.float32)
    src_mask = None
    src_key_padding_mask = None
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": torch.float32,
        "src": src,
        "src_mask": None,
        "src_key_padding_mask": None,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    src = np.random.rand(4, 4, 32).astype(np.float32)
    src_mask = np.triu(np.ones((4, 4))).astype(bool)
    src_key_padding_mask = None
    input_dict = {
        "d_model": 32,
        "nhead": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "activation": "relu",
        "layer_norm_eps": 1e-12,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": torch.float32,
        "src": src,
        "src_mask": src_mask,
        "src_key_padding_mask": None,
        "is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.TransformerEncoderLayer"] = transformer_encoder_layer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.TransformerEncoderLayer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TransformerEncoderLayer'.")

check_valid('torch.nn.TransformerEncoderLayer', generated_inputs['torch.nn.TransformerEncoderLayer'], lib="torch", suffix=0)
