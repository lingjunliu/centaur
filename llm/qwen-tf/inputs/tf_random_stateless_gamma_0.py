
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []
    
    # Input 1: Basic case with shape [2, 3], seed [1, 2], alpha [1.0, 2.0], beta [3.0, 4.0]
    shape = np.array([2, 3], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    alpha = np.array([1.0, 2.0], dtype=np.float32)
    beta = np.array([3.0, 4.0], dtype=np.float32)
    dtype = np.float32
    name = "test"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Shape [5, 4], seed [10, 20], alpha [0.5], beta [2.0]
    shape = np.array([5, 4], dtype=np.int32)
    seed = np.array([10, 20], dtype=np.int32)
    alpha = np.array([0.5], dtype=np.float32)
    beta = np.array([2.0], dtype=np.float32)
    dtype = np.float32
    name = "test2"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Shape [10], seed [5, 15], alpha [1.0], beta [2.0]
    shape = np.array([10], dtype=np.int32)
    seed = np.array([5, 15], dtype=np.int32)
    alpha = np.array([1.0], dtype=np.float32)
    beta = np.array([2.0], dtype=np.float32)
    dtype = np.float64
    name = "test3"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Shape [3, 2], seed [7, 8], alpha [0.1], beta [1.5]
    shape = np.array([3, 2], dtype=np.int32)
    seed = np.array([7, 8], dtype=np.int32)
    alpha = np.array([0.1], dtype=np.float32)
    beta = np.array([1.5], dtype=np.float32)
    dtype = np.float32
    name = "test4"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Shape [7, 5, 2], seed [1, 2], alpha [0.5, 1.0], beta [2.0, 3.0]
    shape = np.array([7, 5, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    alpha = np.array([0.5, 1.0], dtype=np.float32)
    beta = np.array([2.0, 3.0], dtype=np.float32)
    dtype = np.float64
    name = "test5"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Shape [1, 2], seed [3, 4], alpha [1.5], beta [2.0]
    shape = np.array([1, 2], dtype=np.int32)
    seed = np.array([3, 4], dtype=np.int32)
    alpha = np.array([1.5], dtype=np.float32)
    beta = np.array([2.0], dtype=np.float32)
    dtype = np.float32
    name = "test6"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Shape [10, 2], seed [5, 10], alpha [0.1], beta [0.5]
    shape = np.array([10, 2], dtype=np.int32)
    seed = np.array([5, 10], dtype=np.int32)
    alpha = np.array([0.1], dtype=np.float32)
    beta = np.array([0.5], dtype=np.float32)
    dtype = np.float64
    name = "test7"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Shape [3, 3], seed [12, 34], alpha [0.5, 1.0, 2.0], beta [1.0, 2.0, 3.0]
    shape = np.array([3, 3], dtype=np.int32)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    beta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dtype = np.float32
    name = "test8"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Shape [2, 3, 4], seed [1, 2], alpha [0.5], beta [1.0]
    shape = np.array([2, 3, 4], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    alpha = np.array([0.5], dtype=np.float32)
    beta = np.array([1.0], dtype=np.float32)
    dtype = np.float64
    name = "test9"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Shape [3, 2], seed [1, 2], alpha [0.5], beta [2.0]
    shape = np.array([3, 2], dtype=np.int32)
    seed = np.array([1, 2], dtype=np.int32)
    alpha = np.array([0.5], dtype=np.float32)
    beta = np.array([2.0], dtype=np.float32)
    dtype = np.float32
    name = "test10"
    
    input_dict = {
        "shape": shape,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "dtype": dtype,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.random.stateless_gamma"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma'], lib="tf", suffix=0)
