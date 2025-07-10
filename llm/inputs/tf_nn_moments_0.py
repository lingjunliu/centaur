
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_nn_moments_inputs():
    list_of_inputs = []

    # Input 1
    x = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    axes = [0]
    shift = tf.constant(np.array(0.0, dtype=np.float32))
    keepdims = False
    name = "moments_1"
    input_dict = {"x": x, "axes": axes, "shift": shift, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    axes = [0]
    shift = tf.constant(np.array(0.0, dtype=np.float32))
    keepdims = False
    name = "moments_2"
    input_dict = {"x": x, "axes": axes, "shift": shift, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    axes = [1]
    shift = tf.constant(np.array(0.0, dtype=np.float32))
    keepdims = False
    name = "moments_3"
    input_dict = {"x": x, "axes": axes, "shift": shift, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = tf.constant(np.array([[1, 2], [3, 4]], dtype=np.float32))
    axes = [0, 1]
    shift = tf.constant(np.array(0.0, dtype=np.float32))
    keepdims = False
    name = "moments_4"
    input_dict = {"x": x, "axes": axes, "shift": shift, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))
    axes = [0]
    shift = tf.constant(np.array(0.0, dtype=np.float32))
    keepdims = False
    name = "moments_5"
    input_dict = {"x": x, "axes": axes, "shift": shift, "keepdims": keepdims, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.nn.moments"] = tf_nn_moments_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.nn.moments' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.moments'.")

check_valid('tf.nn.moments', generated_inputs['tf.nn.moments'], lib="tf", suffix=0)
