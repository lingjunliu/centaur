
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_rsqrt_inputs():
    list_of_inputs = []

    # Input 1: float32, positive values
    x = tf.constant(np.array([1.0, 4.0, 9.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, includes zero and positive values
    x = tf.constant(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    name = "sqrt_op"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, positive values, multi-dimensional
    x = tf.constant(np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, includes zero and positive values, multi-dimensional
    x = tf.constant(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
    name = "sqrt_op2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: half, positive values
    x = tf.constant(np.array([1.0, 4.0, 9.0], dtype=np.float16))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half, includes zero and positive values
    x = tf.constant(np.array([0.0, 1.0, 2.0], dtype=np.float16))
    name = "sqrt_op4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32, larger values
    x = tf.constant(np.array([100.0, 10000.0, 1000000.0], dtype=np.float32))
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64, different shape
    x = tf.constant(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64))
    name = "sqrt_op5"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
input_list = tf_math_rsqrt_inputs()
for input_dict in input_list:
    x = input_dict['x']
    if isinstance(x, tf.Tensor):
        input_dict['x'] = x.numpy()
    
generated_inputs["tf.math.rsqrt"] = input_list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.rsqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.rsqrt'.")

check_valid('tf.math.rsqrt', generated_inputs['tf.math.rsqrt'], lib="tf", suffix=0)
