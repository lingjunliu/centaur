
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def transformer_inputs():
    list_of_inputs = []

    # Input 1
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    activation = "relu"
    custom_encoder = None
    custom_decoder = None
    layer_norm_eps = 1e-05
    batch_first = False
    norm_first = False
    bias = True
    dtype = np.float32
    src = np.random.rand(10, 32, 512).astype(np.float32)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    src_mask = np.random.rand(10, 10).astype(np.float32)
    tgt_mask = np.random.rand(20, 20).astype(np.float32)
    memory_mask = np.random.rand(20, 10).astype(np.float32)
    src_key_padding_mask = np.random.randint(0, 2, (10, 32), dtype=bool)
    tgt_key_padding_mask = np.random.randint(0, 2, (20, 32), dtype=bool)
    memory_key_padding_mask = np.random.randint(0, 2, (10, 32), dtype=bool)
    src_is_causal = False
    tgt_is_causal = False
    memory_is_causal = False

    input_dict = {
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "activation": activation,
        "custom_encoder": custom_encoder,
        "custom_decoder": custom_decoder,
        "layer_norm_eps": layer_norm_eps,
        "batch_first": batch_first,
        "norm_first": norm_first,
        "bias": bias,
        "dtype": dtype,
        "src": src,
        "tgt": tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "memory_mask": memory_mask,
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask,
        "memory_key_padding_mask": memory_key_padding_mask,
        "src_is_causal": src_is_causal,
        "tgt_is_causal": tgt_is_causal,
        "memory_is_causal": memory_is_causal,
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
