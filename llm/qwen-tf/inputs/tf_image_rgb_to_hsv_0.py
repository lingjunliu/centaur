
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_rgb_to_hsv_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor with RGB values in [0,1]
    a = tf.constant([[[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]], dtype=tf.float32)
    input_dict = {
        "images": a,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor with RGB values in [0,1]
    b = tf.random.uniform([2, 3, 4, 3], minval=0, maxval=1, dtype=tf.float32)
    input_dict = {
        "images": b,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with RGB values in [0,1]
    c = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
    input_dict = {
        "images": c,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 2D tensor with RGB values in [0,1]
    d = tf.constant([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], dtype=tf.float32)
    input_dict = {
        "images": d,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with RGB values in [0,1] and different dtypes
    f = tf.constant([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=tf.float64)
    input_dict = {
        "images": f,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with RGB values in [0,1] and different dtypes
    g = tf.random.uniform([3, 2, 5, 3], minval=0, maxval=1, dtype=tf.half)
    input_dict = {
        "images": g,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 4D tensor with RGB values in [0,1] and different dtypes
    h = tf.random.uniform([2, 3, 4, 3], minval=0, maxval=1, dtype=tf.float32)
    input_dict = {
        "images": h,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with RGB values in [0,1] and different dtypes
    i = tf.constant([[[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]], [[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]], dtype=tf.float64)
    input_dict = {
        "images": i,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 4D tensor with RGB values in [0,1] and different dtypes
    j = tf.random.uniform([2, 3, 4, 3], minval=0, maxval=1, dtype=tf.float32)
    input_dict = {
        "images": j,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 3D tensor with RGB values in [0,1] and different dtypes
    k = tf.constant([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]], dtype=tf.float64)
    input_dict = {
        "images": k,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.image.rgb_to_hsv"] = tf_image_rgb_to_hsv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.rgb_to_hsv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.rgb_to_hsv'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.rgb_to_hsv', generated_inputs['tf.image.rgb_to_hsv'], lib="tf", suffix=0)
