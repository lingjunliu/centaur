
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_greater_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([5, 4, 6], dtype=np.int32)
    y = np.array([5, 2, 5], dtype=np.int32)
    input_dict = {
        "name": "test1",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    x = np.array([5, 4, 6], dtype=np.float32)
    y = np.array([5, 2, 5], dtype=np.float32)
    input_dict = {
        "name": "test2",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    x = np.array([5.0, 4.0, 6.0], dtype=np.float64)
    y = np.array([5.0, 2.0, 5.0], dtype=np.float64)
    input_dict = {
        "name": "test3",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([5, 2, 5], dtype=np.int16)
    input_dict = {
        "name": "test4",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([5, 2, 5], dtype=np.int8)
    input_dict = {
        "name": "test5",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([5, 2, 5], dtype=np.int64)
    input_dict = {
        "name": "test6",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    x = np.array([1, 2, 3], dtype=np.uint8)
    y = np.array([5, 2, 5], dtype=np.uint8)
    input_dict = {
        "name": "test7",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([5, 2, 5], dtype=np.float16)
    input_dict = {
        "name": "test8",
        "x": x,
        "y": y
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Greater"] = tf_greater_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Greater' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Greater'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Greater', generated_inputs['tf.raw_ops.Greater'], lib="tf", suffix=0)
