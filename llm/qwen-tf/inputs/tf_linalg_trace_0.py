
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_trace_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor with float values
    x = np.array([[1.5, 2.7], [3.1, 4.9]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "trace_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with negative values
    x = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with zero values
    x = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with large values
    x = np.array([[100, 200], [300, 400]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with complex numbers
    x = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    input_dict = {
        "x": x,
        "name": "trace_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 2D tensor with mixed values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with varying shapes
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 2D tensor with mixed values
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "x": x,
        "name": "trace_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.trace"] = tf_linalg_trace_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.trace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.trace'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.trace', generated_inputs['tf.linalg.trace'], lib="tf", suffix=0)
