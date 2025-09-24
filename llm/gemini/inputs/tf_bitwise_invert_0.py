
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_invert_inputs():
    list_of_inputs = []

    # Input 1: uint8
    x = tf.constant(np.array([0, 255, 10, 128], dtype=np.uint8))
    name = "invert_uint8_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int8
    x = tf.constant(np.array([-128, 0, 127, -1], dtype=np.int8))
    name = "invert_int8_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32
    x = tf.constant(np.array([-2147483648, 0, 2147483647, -1], dtype=np.int32))
    name = "invert_int32_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int64
    x = tf.constant(np.array([-9223372036854775808, 0, 9223372036854775807, -1], dtype=np.int64))
    name = "invert_int64_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: multi-dimensional int32
    x = tf.constant(np.array([[-1, 0], [1, -2147483648]], dtype=np.int32))
    name = "invert_int32_2d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int8
    x = tf.constant(np.array([1, 2, 3, 4], dtype=np.int8))
    name = "invert_int8_1d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8 scalar
    x = tf.constant(np.array(5, dtype=np.uint8))
    name = "invert_scalar"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.bitwise.invert"] = tf_bitwise_invert_inputs()

for i in range(len(generated_inputs["tf.bitwise.invert"])):
    generated_inputs["tf.bitwise.invert"][i]["x"] = generated_inputs["tf.bitwise.invert"][i]["x"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.bitwise.invert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.invert'.")

check_valid('tf.bitwise.invert', generated_inputs['tf.bitwise.invert'], lib="tf", suffix=0)
