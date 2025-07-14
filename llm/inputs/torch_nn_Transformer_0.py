
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def transformer_inputs():
    list_of_inputs = []

    # Input 1
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, batch_first=True
    src = np.random.rand(32, 10, 512).astype(np.float32)
    tgt = np.random.rand(32, 20, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": True,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
       "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, with masks
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    src_mask = np.random.rand(10, 10).astype(np.float32)
    tgt_mask = np.random.rand(20, 20).astype(np.float32)
    memory_mask = np.random.rand(20, 10).astype(np.float32)

    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "memory_mask": memory_mask,
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, with key padding masks
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    src_key_padding_mask = np.random.randint(0, 2, size=(32, 10)).astype(np.bool_)
    tgt_key_padding_mask = np.random.randint(0, 2, size=(32, 20)).astype(np.bool_)
    memory_key_padding_mask = np.random.randint(0, 2, size=(32, 10)).astype(np.bool_)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
       "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "memory_key_padding_mask": memory_key_padding_mask,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, causal masks, batch_first = True
    src = np.random.rand(32, 10, 512).astype(np.float32)
    tgt = np.random.rand(32, 20, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": True,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
       "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": True,
        "tgt_is_causal": True,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, different d_model
    src = np.random.rand(10, 32, 256).astype(np.float32)
    tgt = np.random.rand(20, 32, 256).astype(np.float32)
    input_dict = {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
       "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, different nhead
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 16,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, gelu activation
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "gelu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, norm_first = True
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": True,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, bias = False
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": False,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
         "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11, different seq lengths
    src = np.random.rand(5, 32, 512).astype(np.float32)
    tgt = np.random.rand(15, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": np.zeros((0,0)).astype(np.float32) ,
        "tgt_mask":  np.zeros((0,0)).astype(np.float32),
        "memory_mask":  np.zeros((0,0)).astype(np.float32),
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": False,
        "tgt_is_causal": False,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12, src_is_causal = True, tgt_is_causal = True
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    input_dict = {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
        "custom_encoder": None,
        "custom_decoder": None,
        "layer_norm_eps": 1e-05,
        "batch_first": False,
        "norm_first": False,
        "bias": True,
        "dtype": np.dtype('float32'),
        "src": src,
        "tgt": tgt,
        "src_mask": None,
        "tgt_mask": None,
        "memory_mask": None,
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
        "src_is_causal": True,
        "tgt_is_causal": True,
        "memory_is_causal": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Transformer"] = transformer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Transformer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Transformer'.")

check_valid('torch.nn.Transformer', generated_inputs['torch.nn.Transformer'], lib="torch", suffix=0)
