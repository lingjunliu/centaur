
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_pow_inputs():
    list_of_inputs = []

    # Input 1: Basic integer powers
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[2, 3], [2, 1]], dtype=np.int32)
    name = "pow_int"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float powers
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[2.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    name = "pow_float"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative powers
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    y = np.array([[-1.0, -2.0], [-0.5, 1.0]], dtype=np.float32)
    name = "pow_negative"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex powers
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    y = np.array([[2+0j, 0+1j], [1+0j, 2+1j]], dtype=np.complex64)
    name = "pow_complex"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes, but broadcastable
    x = np.array([2, 3, 4], dtype=np.int32)
    y = np.array([2], dtype=np.int32)
    name = "pow_broadcast"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero power
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[0, 0], [0, 0]], dtype=np.int32)
    name = "pow_zero"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One power
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[1, 1], [1, 1]], dtype=np.int32)
    name = "pow_one"
    input_dict = {"x": tf.convert_to_tensor(x), "y": tf.convert_to_tensor(y), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.pow'.")

check_valid('tf.math.pow', generated_inputs['tf.math.pow'], lib="tf", suffix=0)
