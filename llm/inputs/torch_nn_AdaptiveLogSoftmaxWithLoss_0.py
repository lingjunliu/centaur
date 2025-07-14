
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_log_softmax_with_loss_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    in_features = 10
    n_classes = 100
    cutoffs = [20, 50]
    div_value = 4.0
    head_bias = False
    dtype = np.float32
    input_tensor = np.random.randn(5, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(5,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different in_features and n_classes
    in_features = 20
    n_classes = 500
    cutoffs = [100, 250, 400]
    div_value = 2.0
    head_bias = True
    dtype = np.float64
    input_tensor = np.random.randn(10, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(10,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  Single input
    in_features = 5
    n_classes = 20
    cutoffs = [5, 10]
    div_value = 2.0
    head_bias = False
    dtype = np.float32
    input_tensor = np.random.randn(in_features).astype(dtype)
    target_tensor = np.array(3).astype(np.int64)

    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Different cutoffs
    in_features = 15
    n_classes = 300
    cutoffs = [50, 150]
    div_value = 3.0
    head_bias = True
    dtype = np.float32
    input_tensor = np.random.randn(8, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(8,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger div_value
    in_features = 8
    n_classes = 60
    cutoffs = [10, 20, 30]
    div_value = 8.0
    head_bias = False
    dtype = np.float64
    input_tensor = np.random.randn(4, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(4,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No head bias
    in_features = 12
    n_classes = 150
    cutoffs = [30, 75]
    div_value = 5.0
    head_bias = False
    dtype = np.float32
    input_tensor = np.random.randn(6, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(6,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Small numbers for everything
    in_features = 2
    n_classes = 10
    cutoffs = [2]
    div_value = 2.0
    head_bias = True
    dtype = np.float32
    input_tensor = np.random.randn(3, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(3,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different target type (int32) - changed to int64 to avoid potential issues
    in_features = 10
    n_classes = 100
    cutoffs = [20, 50]
    div_value = 4.0
    head_bias = False
    dtype = np.float32
    input_tensor = np.random.randn(5, in_features).astype(dtype)
    target_tensor = np.random.randint(0, n_classes, size=(5,)).astype(np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Target = 0
    in_features = 10
    n_classes = 100
    cutoffs = [20, 50]
    div_value = 4.0
    head_bias = False
    dtype = np.float32
    input_tensor = np.random.randn(5, in_features).astype(dtype)
    target_tensor = np.zeros(5, dtype=np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
     
    # Input 10: Max target value
    in_features = 10
    n_classes = 100
    cutoffs = [20, 50]
    div_value = 4.0
    head_bias = False
    dtype = np.float32
    input_tensor = np.random.randn(5, in_features).astype(dtype)
    target_tensor = np.full(5, n_classes -1, dtype=np.int64)
    input_dict = {
        "in_features": int(in_features),
        "n_classes": int(n_classes),
        "cutoffs": cutoffs,
        "div_value": float(div_value),
        "head_bias": bool(head_bias),
        "dtype": torch.float32 if dtype == np.float32 else torch.float64,
        "input": input_tensor,
        "target": target_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AdaptiveLogSoftmaxWithLoss"] = adaptive_log_softmax_with_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AdaptiveLogSoftmaxWithLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveLogSoftmaxWithLoss'.")

check_valid('torch.nn.AdaptiveLogSoftmaxWithLoss', generated_inputs['torch.nn.AdaptiveLogSoftmaxWithLoss'], lib="torch", suffix=0)
