
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def embeddingbag_inputs():
    list_of_inputs = []

    # Input 1
    num_embeddings = 10
    embedding_dim = 5
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "mean"
    sparse = False
    include_last_offset = False
    padding_idx = 0
    dtype = np.float32
    input = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    offsets = np.array([0, 4], dtype=np.int64)
    per_sample_weights = None

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    num_embeddings = 5
    embedding_dim = 3
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = True
    mode = "sum"
    sparse = True
    include_last_offset = False
    padding_idx = None
    dtype = np.float64
    input = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float64,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    num_embeddings = 12
    embedding_dim = 7
    max_norm = 0.5
    norm_type = 3.0
    scale_grad_by_freq = False
    mode = "max"
    sparse = False
    include_last_offset = False
    padding_idx = 1
    dtype = np.float32
    input = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    offsets = np.array([0, 2, 5, 7], dtype=np.int64)
    per_sample_weights = None
    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    num_embeddings = 8
    embedding_dim = 4
    max_norm = 2.0
    norm_type = 1.0
    scale_grad_by_freq = True
    mode = "mean"
    sparse = False
    include_last_offset = True
    padding_idx = 3
    dtype = np.float64
    input = np.array([0, 1, 2, 4, 5, 6, 7], dtype=np.int64)
    offsets = np.array([0, 3, 5, 7, 7], dtype=np.int64)
    per_sample_weights = None
    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float64,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    num_embeddings = 15
    embedding_dim = 6
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "sum"
    sparse = True
    include_last_offset = False
    padding_idx = 0
    dtype = np.float32
    input = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    per_sample_weights = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, 2D input
    num_embeddings = 10
    embedding_dim = 5
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "mean"
    sparse = False
    include_last_offset = False
    padding_idx = 0
    dtype = np.float32
    input = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
    offsets = None
    per_sample_weights = None

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, padding_idx = None
    num_embeddings = 10
    embedding_dim = 5
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "mean"
    sparse = False
    include_last_offset = False
    padding_idx = None
    dtype = np.float32
    input = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    offsets = np.array([0, 4], dtype=np.int64)
    per_sample_weights = None

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, mode = "max"
    num_embeddings = 10
    embedding_dim = 5
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "max"
    sparse = False
    include_last_offset = False
    padding_idx = 0
    dtype = np.float32
    input = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    offsets = np.array([0, 4], dtype=np.int64)
    per_sample_weights = None

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, include_last_offset=True, offsets covers the entire input
    num_embeddings = 6
    embedding_dim = 2
    max_norm = None
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "mean"
    sparse = False
    include_last_offset = True
    padding_idx = None
    dtype = np.float32
    input = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    offsets = np.array([0, 2, 4, 6], dtype=np.int64)
    per_sample_weights = None

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, empty bags in offsets
    num_embeddings = 10
    embedding_dim = 5
    max_norm = 1.0
    norm_type = 2.0
    scale_grad_by_freq = False
    mode = "mean"
    sparse = False
    include_last_offset = False
    padding_idx = 0
    dtype = np.float32
    input = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    offsets = np.array([0, 0, 4, 4], dtype=np.int64) # Empty bags at 0 and 4
    per_sample_weights = None

    input_dict = {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "max_norm": max_norm,
        "norm_type": norm_type,
        "scale_grad_by_freq": scale_grad_by_freq,
        "mode": mode,
        "sparse": sparse,
        "include_last_offset": include_last_offset,
        "padding_idx": padding_idx,
        "dtype": torch.float32,
        "input": input,
        "offsets": offsets,
        "per_sample_weights": per_sample_weights,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.EmbeddingBag"] = embeddingbag_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.EmbeddingBag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.EmbeddingBag'.")

check_valid('torch.nn.EmbeddingBag', generated_inputs['torch.nn.EmbeddingBag'], lib="torch", suffix=0)
