
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def MultiheadAttention_inputs():
    list_of_inputs = []

    embed_dim = 16
    num_heads = 4

    query = torch.randn(10, 3, embed_dim).float().numpy()
    key = torch.randn(10, 3, embed_dim).float().numpy()
    value = torch.randn(10, 3, embed_dim).float().numpy()

    input_dict = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": 0.0,
        "bias": True,
        "add_bias_kv": False,
        "add_zero_attn": False,
        "kdim": None,
        "vdim": None,
        "batch_first": True,
        "dtype": torch.float32,
        "query": query,
        "key": key,
        "value": value,
        "key_padding_mask": None,
        "need_weights": True,
        "attn_mask": None,
        "average_attn_weights": True,
        "is_causal": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    embed_dim = 32
    num_heads = 8
    query = torch.randn(5, 20, embed_dim).float().numpy()
    key = torch.randn(5, 10, embed_dim).float().numpy()
    value = torch.randn(5, 10, embed_dim).float().numpy()
    key_padding_mask = torch.randint(0, 2, (5, 10)).bool().numpy()

    input_dict = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": 0.1,
        "bias": False,
        "add_bias_kv": True,
        "add_zero_attn": True,
        "kdim": None,
        "vdim": None,
        "batch_first": True,
        "dtype": torch.float32,
        "query": query,
        "key": key,
        "value": value,
        "key_padding_mask": key_padding_mask,
        "need_weights": False,
        "attn_mask": None,
        "average_attn_weights": False,
        "is_causal": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    embed_dim = 64
    num_heads = 4
    query = torch.randn(20, 5, embed_dim).float().numpy()
    key = torch.randn(10, 5, embed_dim).float().numpy()
    value = torch.randn(10, 5, embed_dim).float().numpy()
    attn_mask = torch.rand(20, 10).float().numpy()

    input_dict = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": 0.2,
        "bias": True,
        "add_bias_kv": False,
        "add_zero_attn": False,
        "kdim": 64,
        "vdim": 64,
        "batch_first": False,
        "dtype": torch.float32,
        "query": query,
        "key": key,
        "value": value,
        "key_padding_mask": None,
        "need_weights": True,
        "attn_mask": attn_mask,
        "average_attn_weights": True,
        "is_causal": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    embed_dim = 128
    num_heads = 8
    query = torch.randn(15, 1, embed_dim).float().numpy()
    key = torch.randn(7, 1, embed_dim).float().numpy()
    value = torch.randn(7, 1, embed_dim).float().numpy()
    attn_mask = torch.rand(num_heads, 15, 7).float().numpy()

    input_dict = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": 0.0,
        "bias": False,
        "add_bias_kv": True,
        "add_zero_attn": False,
        "kdim": 128,
        "vdim": 128,
        "batch_first": False,
        "dtype": torch.float32,
        "query": query,
        "key": key,
        "value": value,
        "key_padding_mask": None,
        "need_weights": False,
        "attn_mask": attn_mask,
        "average_attn_weights": False,
        "is_causal": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    embed_dim = 256
    num_heads = 16
    query = torch.randn(1, 3, embed_dim).float().numpy()
    key = torch.randn(1, 3, embed_dim).float().numpy()
    value = torch.randn(1, 3, embed_dim).float().numpy()
    attn_mask = torch.tril(torch.ones(3, 3)).float().numpy()

    input_dict = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "dropout": 0.1,
        "bias": True,
        "add_bias_kv": False,
        "add_zero_attn": True,
        "kdim": 256,
        "vdim": 256,
        "batch_first": True,
        "dtype": torch.float32,
        "query": query,
        "key": key,
        "value": value,
        "key_padding_mask": None,
        "need_weights": True,
        "attn_mask": attn_mask,
        "average_attn_weights": True,
        "is_causal": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MultiheadAttention"] = MultiheadAttention_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MultiheadAttention' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiheadAttention'.")

check_valid('torch.nn.MultiheadAttention', generated_inputs['torch.nn.MultiheadAttention'], lib="torch")
