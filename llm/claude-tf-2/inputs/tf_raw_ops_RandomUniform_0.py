
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_RandomUniform_inputs():
    list_of_inputs = []
    
    input_dict = {
        "shape": np.array([10], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 42,
        "seed2": 1,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([5, 5], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 42,
        "seed2": 123,
        "name": "random_uniform_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 3, 4], dtype=np.int64),
        "dtype": tf.float64,
        "seed": 1,
        "seed2": 2,
        "name": "random_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 2, 2, 2], dtype=np.int32),
        "dtype": tf.half,
        "seed": 100,
        "seed2": 200,
        "name": "random_half"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([100, 50], dtype=np.int64),
        "dtype": tf.bfloat16,
        "seed": 999,
        "seed2": 888,
        "name": "random_bfloat16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([1], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 5,
        "seed2": 10,
        "name": "single_element"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 3, 4, 5, 6], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 7,
        "seed2": 14,
        "name": "random_5d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([8, 8], dtype=np.int64),
        "dtype": tf.float64,
        "seed": 50,
        "seed2": 60,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([1000], dtype=np.int32),
        "dtype": tf.float32,
        "seed": 12345,
        "seed2": 67890,
        "name": "large_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([4, 5, 6], dtype=np.int32),
        "dtype": tf.float64,
        "seed": 33,
        "seed2": 44,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.RandomUniform"] = tf_raw_ops_RandomUniform_inputs()

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
