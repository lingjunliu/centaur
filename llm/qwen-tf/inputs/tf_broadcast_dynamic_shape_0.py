
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_broadcast_dynamic_shape_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 1, 3], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    shape_x = np.array([2, 3, 4], dtype=np.int32)
    shape_y = np.array([2, 3, 4], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    shape_x = np.array([1, 1, 1], dtype=np.int32)
    shape_y = np.array([5, 6, 7], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    shape_x = np.array([10, 20], dtype=np.int32)
    shape_y = np.array([10, 20], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    shape_x = np.array([1, 2, 3], dtype=np.int32)
    shape_y = np.array([5, 6, 7, 8], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    shape_x = np.array([10, 20, 30], dtype=np.int32)
    shape_y = np.array([10, 20], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    shape_x = np.array([5, 6, 7], dtype=np.int32)
    shape_y = np.array([10, 20, 30, 40], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    shape_x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shape_y = np.array([10, 20], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    shape_x = np.array([5, 6, 7], dtype=np.int32)
    shape_y = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    shape_x = np.array([1, 2, 3, 4], dtype=np.int32)
    shape_y = np.array([5, 6, 7], dtype=np.int32)
    input_dict = {
        "shape_x": shape_x,
        "shape_y": shape_y
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.broadcast_dynamic_shape"] = tf_broadcast_dynamic_shape_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.broadcast_dynamic_shape' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.broadcast_dynamic_shape'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.broadcast_dynamic_shape', generated_inputs['tf.broadcast_dynamic_shape'], lib="tf", suffix=0)
