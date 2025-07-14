
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embedding_inputs():
    list_of_inputs = []

    # Input 1
    input = np.array([1, 2, 3], dtype=np.int64)
    num_embeddings = 5
    embedding_dim = 4
    padding_idx = 0
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.array([[1, 2], [3, 4]], dtype=np.int64)
    num_embeddings = 6
    embedding_dim = 3
    padding_idx = 1
    max_norm = 2.0
    norm_type = 1.0
    scale_grad_by_freq = True
    sparse = True

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    num_embeddings = 7
    embedding_dim = 2
    padding_idx = 2
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = np.array([[1,0,1],[2,3,4]], dtype=np.int64)
    num_embeddings = 8
    embedding_dim = 5
    padding_idx = 7
    max_norm = 0.5
    norm_type = 3.0
    scale_grad_by_freq = True
    sparse = True

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float16,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.array([5, 6, 0, 1, 2], dtype=np.int64)
    num_embeddings = 10
    embedding_dim = 1
    padding_idx = 5
    max_norm = 1.5
    norm_type = 0.5
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input = np.array([[4, 5], [2, 3]], dtype=np.int64)
    num_embeddings = 14
    embedding_dim = 6
    padding_idx = 13
    max_norm = 2.5
    norm_type = 1.5
    scale_grad_by_freq = True
    sparse = True

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.array([4, 5, 6, 0, 1], dtype=np.int64)
    num_embeddings = 20
    embedding_dim = 7
    padding_idx = 19
    max_norm = None
    norm_type = 2.5
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = np.array([[1,0,2],[3,4,5]], dtype=np.int64)
    num_embeddings = 22
    embedding_dim = 8
    padding_idx = 21
    max_norm = 0.7
    norm_type = 3.5
    scale_grad_by_freq = True
    sparse = True

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float16,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.array([4, 5, 6, 7, 8], dtype=np.int64)
    num_embeddings = 25
    embedding_dim = 9
    padding_idx = 24
    max_norm = 1.7
    norm_type = 0.7
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.array([[5, 6], [7, 8]], dtype=np.int64)
    num_embeddings = 30
    embedding_dim = 10
    padding_idx = 29
    max_norm = 2.7
    norm_type = 1.7
    scale_grad_by_freq = True
    sparse = True

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input = np.array([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=np.int64)
    num_embeddings = 31
    embedding_dim = 11
    padding_idx = 30
    max_norm = 3.0
    norm_type = 2.0
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input = np.array([0], dtype=np.int64)
    num_embeddings = 2
    embedding_dim = 3
    padding_idx = 0
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = False
    sparse = False

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Embedding"] = embedding_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Embedding' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Embedding'.")

check_valid('torch.nn.Embedding', generated_inputs['torch.nn.Embedding'], lib="torch", suffix=0)
