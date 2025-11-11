
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def generate_random_uniform_inputs():
    list_of_inputs = []
    
    # Input 1: shape = [2, 3], dtype = float32, seed = 1, seed2 = 2, name = "test"
    shape = np.array([2, 3], dtype=np.int32)
    dtype = np.float32
    seed = 1
    seed2 = 2
    name = "test"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: shape = [5], dtype = float64, seed = 0, seed2 = 0, name = "test2"
    shape = np.array([5], dtype=np.int32)
    dtype = np.float64
    seed = 0
    seed2 = 0
    name = "test2"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: shape = [3, 4, 5], dtype = bfloat16, seed = 42, seed2 = 100, name = "test3"
    shape = np.array([3, 4, 5], dtype=np.int32)
    dtype = np.float16
    seed = 42
    seed2 = 100
    name = "test3"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: shape = [1, 1], dtype = half, seed = 99, seed2 = 88, name = "test4"
    shape = np.array([1, 1], dtype=np.int32)
    dtype = np.float16
    seed = 99
    seed2 = 88
    name = "test4"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: shape = [10, 1], dtype = float32, seed = 1000, seed2 = 2000, name = "test5"
    shape = np.array([10, 1], dtype=np.int32)
    dtype = np.float32
    seed = 1000
    seed2 = 2000
    name = "test5"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: shape = [7, 8, 9, 10], dtype = float64, seed = 555, seed2 = 666, name = "test6"
    shape = np.array([7, 8, 9, 10], dtype=np.int32)
    dtype = np.float64
    seed = 555
    seed2 = 666
    name = "test6"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: shape = [0], dtype = float32, seed = 123, seed2 = 456, name = "test7"
    shape = np.array([0], dtype=np.int32)
    dtype = np.float32
    seed = 123
    seed2 = 456
    name = "test7"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: shape = [100], dtype = bfloat16, seed = 789, seed2 = 321, name = "test8"
    shape = np.array([100], dtype=np.int32)
    dtype = np.float16
    seed = 789
    seed2 = 321
    name = "test8"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: shape = [2, 3, 4], dtype = half, seed = 0, seed2 = 1, name = "test9"
    shape = np.array([2, 3, 4], dtype=np.int32)
    dtype = np.float16
    seed = 0
    seed2 = 1
    name = "test9"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: shape = [5, 5], dtype = float32, seed = 42, seed2 = 100, name = "test10"
    shape = np.array([5, 5], dtype=np.int32)
    dtype = np.float32
    seed = 42
    seed2 = 100
    name = "test10"
    
    input_dict = {
        'seed': seed,
        'seed2': seed2,
        'name': name,
        'shape': shape,
        'dtype': dtype
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = generate_random_uniform_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomUniform' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomUniform'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomUniform', generated_inputs['tf.raw_ops.RandomUniform'], lib="tf", suffix=0)
