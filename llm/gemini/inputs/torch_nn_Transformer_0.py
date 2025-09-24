
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def transformer_inputs():
    list_of_inputs = []

    def create_tensors(S, T, N, E, batch_first, dtype):
        torch_dtype = getattr(torch, dtype.__name__)
        if batch_first:
            src = torch.rand((N, S, E), dtype=torch_dtype).numpy()
            tgt = torch.rand((N, T, E), dtype=torch_dtype).numpy()
        else:
            src = torch.rand((S, N, E), dtype=torch_dtype).numpy()
            tgt = torch.rand((T, N, E), dtype=torch_dtype).numpy()
        return src, tgt

    def create_base_dict(**kwargs):
        empty_tensor = torch.empty(0).numpy()
        base = {
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'activation': 'relu',
            'custom_encoder': [],
            'custom_decoder': [],
            'layer_norm_eps': 1e-5,
            'batch_first': False,
            'norm_first': False,
            'bias': True,
            'dtype': np.float32,
            'src_mask': empty_tensor,
            'tgt_mask': empty_tensor,
            'memory_mask': empty_tensor,
            'src_key_padding_mask': empty_tensor,
            'tgt_key_padding_mask': empty_tensor,
            'memory_key_padding_mask': empty_tensor,
            'src_is_causal': False,
            'tgt_is_causal': False,
            'memory_is_causal': False,
        }
        base.update(kwargs)
        return base

    # Input 1: Basic case, batch_first=False
    d_model, nhead = 64, 4
    S, T, N = 10, 20, 8
    src, tgt = create_tensors(S, T, N, d_model, False, np.float32)
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, dim_feedforward=128, src=src, tgt=tgt)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: batch_first=True and 'gelu' activation
    d_model, nhead = 128, 8
    S, T, N = 12, 18, 4
    src, tgt = create_tensors(S, T, N, d_model, True, np.float32)
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu', src=src, tgt=tgt)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With a causal target mask (tgt_mask)
    d_model, nhead = 32, 2
    S, T, N = 8, 16, 6
    src, tgt = create_tensors(S, T, N, d_model, False, np.float32)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).numpy()
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, dim_feedforward=64, src=src, tgt=tgt, tgt_mask=tgt_mask)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With boolean key padding masks
    d_model, nhead = 64, 4
    S, T, N = 10, 20, 8
    src, tgt = create_tensors(S, T, N, d_model, True, np.float32)
    src_key_padding_mask = (torch.rand(N, S) > 0.5).numpy()
    tgt_key_padding_mask = (torch.rand(N, T) > 0.5).numpy()
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, batch_first=True, src=src, tgt=tgt,
                                  src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With all types of masks
    d_model, nhead = 128, 8
    S, T, N = 15, 25, 4
    src, tgt = create_tensors(S, T, N, d_model, True, np.float32)
    src_mask = (torch.rand(S, S) > 0.5).numpy()
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).numpy()
    memory_mask = torch.randn(T, S).numpy()
    src_key_padding_mask = (torch.rand(N, S) > 0.5).numpy()
    tgt_key_padding_mask = (torch.rand(N, T) > 0.5).numpy()
    memory_key_padding_mask = src_key_padding_mask.copy()
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, batch_first=True, src=src, tgt=tgt,
                                  src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: norm_first=True and bias=False
    d_model, nhead = 256, 8
    S, T, N = 10, 20, 16
    src, tgt = create_tensors(S, T, N, d_model, False, np.float32)
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, norm_first=True, bias=False, src=src, tgt=tgt)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float64 dtype and different layers
    d_model, nhead = 256, 4
    S, T, N = 10, 10, 2
    dtype = np.float64
    src, tgt = create_tensors(S, T, N, d_model, True, dtype)
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, num_encoder_layers=4, num_decoder_layers=8,
                                  dropout=0.5, batch_first=True, dtype=dtype, src=src, tgt=tgt)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Using is_causal flags to auto-generate masks
    d_model, nhead = 64, 4
    S, T, N = 8, 8, 4
    src, tgt = create_tensors(S, T, N, d_model, True, np.float32)
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, batch_first=True, src=src, tgt=tgt,
                                  src_is_causal=True, tgt_is_causal=True)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: With batched float masks
    d_model, nhead = 64, 4
    S, T, N = 10, 15, 2
    src, tgt = create_tensors(S, T, N, d_model, True, np.float32)
    src_mask = torch.rand(N * nhead, S, S).numpy()
    tgt_mask = torch.rand(N * nhead, T, T).numpy()
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, batch_first=True, src=src, tgt=tgt,
                                  src_mask=src_mask, tgt_mask=tgt_mask)
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: No dropout and minimal layers
    d_model, nhead = 128, 8
    S, T, N = 10, 10, 5
    src, tgt = create_tensors(S, T, N, d_model, False, np.float32)
    input_dict = create_base_dict(d_model=d_model, nhead=nhead, num_encoder_layers=1, num_decoder_layers=1,
                                  dropout=0.0, src=src, tgt=tgt)
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.Transformer"] = transformer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Transformer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Transformer'.")

check_valid('torch.nn.Transformer', generated_inputs['torch.nn.Transformer'], lib="torch", suffix=0)
