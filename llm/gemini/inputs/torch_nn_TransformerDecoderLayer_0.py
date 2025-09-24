
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def generate_transformerdecoderlayer_inputs():
    inputs = []

    # Input 1
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)
    tgt_mask = np.triu(np.ones((20, 20), dtype=bool), k=1)
    memory_mask = np.zeros((20, 10), dtype=bool)
    tgt_key_padding_mask = np.zeros((32, 20), dtype=bool)
    memory_key_padding_mask = np.zeros((32, 10), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })

    # Input 2 (batch_first=True)
    tgt = np.random.rand(32, 20, 512).astype(np.float32)
    memory = np.random.rand(32, 10, 512).astype(np.float32)
    tgt_mask = np.triu(np.ones((20, 20), dtype=bool), k=1)
    memory_mask = np.zeros((20, 10), dtype=bool)
    tgt_key_padding_mask = np.zeros((32, 20), dtype=bool)
    memory_key_padding_mask = np.zeros((32, 10), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': True,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })

    # Input 3 (different d_model, nhead, dim_feedforward)
    tgt = np.random.rand(15, 64, 256).astype(np.float32)
    memory = np.random.rand(8, 64, 256).astype(np.float32)
    tgt_mask = np.triu(np.ones((15, 15), dtype=bool), k=1)
    memory_mask = np.zeros((15, 8), dtype=bool)
    tgt_key_padding_mask = np.zeros((64, 15), dtype=bool)
    memory_key_padding_mask = np.zeros((64, 8), dtype=bool)

    inputs.append({
        'd_model': 256,
        'nhead': 4,
        'dim_feedforward': 1024,
        'dropout': 0.2,
        'activation': 'gelu',
        'layer_norm_eps': 1e-06,
        'batch_first': False,
        'norm_first': True,
        'bias': False,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask,
        'tgt_is_causal': True,
        'memory_is_causal': True
    })

    # Input 4 (different shapes, batch_first=True)
    tgt = np.random.rand(64, 15, 256).astype(np.float32)
    memory = np.random.rand(64, 8, 256).astype(np.float32)
    tgt_mask = np.triu(np.ones((15, 15), dtype=bool), k=1)
    memory_mask = np.zeros((15, 8), dtype=bool)
    tgt_key_padding_mask = np.zeros((64, 15), dtype=bool)
    memory_key_padding_mask = np.zeros((64, 8), dtype=bool)

    inputs.append({
        'd_model': 256,
        'nhead': 4,
        'dim_feedforward': 1024,
        'dropout': 0.2,
        'activation': 'gelu',
        'layer_norm_eps': 1e-06,
        'batch_first': True,
        'norm_first': True,
        'bias': False,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask,
        'tgt_is_causal': True,
        'memory_is_causal': True
    })

    # Input 5 (tgt_mask and memory_mask are None)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)
    tgt_key_padding_mask = np.zeros((32, 20), dtype=bool)
    memory_key_padding_mask = np.zeros((32, 10), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': None,
        'memory_mask': None,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })
    
    # Input 6 (tgt_key_padding_mask and memory_key_padding_mask are None)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)
    tgt_mask = np.triu(np.ones((20, 20), dtype=bool), k=1)
    memory_mask = np.zeros((20, 10), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': None,
        'memory_key_padding_mask': None,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })
    
    # Input 7 (tgt_mask and tgt_key_padding_mask are None)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)
    memory_mask = np.zeros((20, 10), dtype=bool)
    memory_key_padding_mask = np.zeros((32, 10), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': None,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': None,
        'memory_key_padding_mask': memory_key_padding_mask,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })

    # Input 8 (memory_mask and memory_key_padding_mask are None)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)
    tgt_mask = np.triu(np.ones((20, 20), dtype=bool), k=1)
    tgt_key_padding_mask = np.zeros((32, 20), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': None,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': None,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })
    
    # Input 9 (all masks are None)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': None,
        'memory_mask': None,
        'tgt_key_padding_mask': None,
        'memory_key_padding_mask': None,
        'tgt_is_causal': False,
        'memory_is_causal': False
    })
    
    # Input 10 (tgt_is_causal = True, memory_is_causal=True, norm_first=True)
    tgt = np.random.rand(20, 32, 512).astype(np.float32)
    memory = np.random.rand(10, 32, 512).astype(np.float32)
    tgt_mask = np.triu(np.ones((20, 20), dtype=bool), k=1)
    memory_mask = np.zeros((20, 10), dtype=bool)

    inputs.append({
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': True,
        'bias': True,
        'tgt': tgt,
        'memory': memory,
        'tgt_mask': tgt_mask,
        'memory_mask': memory_mask,
        'tgt_key_padding_mask': None,
        'memory_key_padding_mask': None,
        'tgt_is_causal': True,
        'memory_is_causal': True
    })

    return inputs

generated_inputs = {}
generated_inputs["torch.nn.TransformerDecoderLayer"] = generate_transformerdecoderlayer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.TransformerDecoderLayer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TransformerDecoderLayer'.")

check_valid('torch.nn.TransformerDecoderLayer', generated_inputs['torch.nn.TransformerDecoderLayer'], lib="torch", suffix=0)
