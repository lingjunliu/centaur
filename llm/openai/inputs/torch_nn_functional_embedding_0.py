
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def embedding_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long).numpy()
    weight_arr = torch.rand(10, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(0),
        "max_norm": float(1.0),
        "norm_type": float(2.0),
        "scale_grad_by_freq": False,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.tensor([0, 5, 2, 1, 5], dtype=torch.long).numpy()
    weight_arr = torch.randn(6, 2, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(5),
        "max_norm": float(0.5),
        "norm_type": float(1.0),
        "scale_grad_by_freq": True,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 0], [1, 2, 3]]], dtype=torch.long).numpy()
    weight_arr = torch.randn(8, 4, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(7),
        "max_norm": float(2.5),
        "norm_type": float(2.0),
        "scale_grad_by_freq": False,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4 (scalar input)
    input_arr = np.array(3, dtype=np.int64)
    weight_arr = torch.randn(5, 1, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(0),
        "max_norm": float(2.0),
        "norm_type": float(2.0),
        "scale_grad_by_freq": False,
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5 (large vocab, include max index)
    input_arr = torch.tensor([10, 999, 5, 0], dtype=torch.long).numpy()
    weight_arr = torch.randn(1000, 64, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(0),
        "max_norm": float(1.2),
        "norm_type": float(2.0),
        "scale_grad_by_freq": True,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6 (float64 weights, fractional norm_type)
    input_arr = torch.tensor([[[0, 1]], [[2, 1]]], dtype=torch.long).numpy()
    weight_arr = torch.randn(3, 5, dtype=torch.float64).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(1),
        "max_norm": float(3.0),
        "norm_type": float(1.5),
        "scale_grad_by_freq": True,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7 (small max_norm)
    input_arr = torch.tensor([1, 3, 6], dtype=torch.long).numpy()
    weight_arr = torch.randn(7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(6),
        "max_norm": float(0.1),
        "norm_type": float(2.0),
        "scale_grad_by_freq": False,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8 (include padding_idx in input)
    input_arr = torch.tensor([[0, 2, 0], [1, 0, 3]], dtype=torch.long).numpy()
    weight_arr = torch.randn(4, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(0),
        "max_norm": float(5.0),
        "norm_type": float(2.0),
        "scale_grad_by_freq": True,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9 (4D input)
    input_arr = torch.randint(0, 5, (2, 2, 2, 2), dtype=torch.long).numpy()
    weight_arr = torch.randn(5, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(4),
        "max_norm": float(1.0),
        "norm_type": float(1.0),
        "scale_grad_by_freq": False,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (embedding_dim = 1)
    input_arr = torch.tensor([3, 1, 0, 2], dtype=torch.long).numpy()
    weight_arr = torch.randn(4, 1, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(3),
        "max_norm": float(10.0),
        "norm_type": float(2.0),
        "scale_grad_by_freq": True,
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11 (non-square shapes, varied values)
    input_arr = torch.tensor([[2], [1], [4], [0]], dtype=torch.long).numpy()
    weight_arr = torch.randn(6, 7, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "weight": weight_arr,
        "padding_idx": int(5),
        "max_norm": float(4.2),
        "norm_type": float(2.0),
        "scale_grad_by_freq": False,
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.embedding"] = embedding_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.embedding' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.embedding'.")


check_valid('torch.nn.functional.embedding', generated_inputs['torch.nn.functional.embedding'], lib="torch", suffix=0)
