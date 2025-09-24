
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_sign_inputs():
    list_of_inputs = []

    # Input 1: float32
    x = tf.constant(np.array([-1.0, 0.0, 2.0, -3.5]), dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64
    x = tf.constant(np.array([1.5, -2.7, 0.0, 3.14159]), dtype=np.float64)
    name = "sign_float64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    x = tf.constant(np.array([-5, 0, 10, -2]), dtype=np.int32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64
    x = tf.constant(np.array([100, -200, 0, 50]), dtype=np.int64)
    name = "sign_int64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64
    x = tf.constant(np.array([1 + 1j, -1 - 1j, 0 + 0j, 2 - 2j]), dtype=np.complex64)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = tf.constant(np.array([0 + 1j, -2 + 0j, 0 - 0j, 1 - 2j]), dtype=np.complex128)
    name = "sign_complex128"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: bfloat16 (removed to avoid error, as bfloat16 might not be directly convertible to numpy)
    #x = tf.constant(np.array([-1.0, 0.0, 2.0, -3.5]).astype(np.float16), dtype=tf.bfloat16)
    #name = None
    #input_dict = {"x": x, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half (removed to avoid error, as float16 might not be directly convertible to numpy)
    #x = tf.constant(np.array([1.5, -2.7, 0.0, 3.14159]).astype(np.float16), dtype=tf.float16)
    #name = "sign_half"
    #input_dict = {"x": x, "name": name}
    #list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimension float32
    x = tf.constant(np.array([[[1.0, -2.0], [0.0, 3.0]], [[-4.0, 5.0], [6.0, -7.0]]]), dtype=np.float32)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher dimension complex128
    x = tf.constant(np.array([[[1 + 1j, -2 - 2j], [0 + 0j, 3 + 3j]], [[-4 - 4j, 5 + 5j], [6 + 6j, -7 - 7j]]]), dtype=np.complex128)
    name = None
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_math_sign_inputs()
for i in range(len(inputs)):
  inputs[i]['x'] = inputs[i]['x'].numpy()

generated_inputs["tf.math.sign"] = inputs

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.sign'.")

check_valid('tf.math.sign', generated_inputs['tf.math.sign'], lib="tf", suffix=0)
