
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_conj_inputs():
    list_of_inputs = []

    # Input 1: Complex numbers
    x = tf.constant([1 + 1j, 2 + 2j, 3 + 3j]).numpy()
    name = "complex_conj_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Real numbers
    x = tf.constant([1.0, 2.0, 3.0]).numpy()
    name = "real_conj_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Mixed complex and real
    x = tf.constant([1 + 1j, 2.0, 3 + 3j]).numpy()
    name = "mixed_conj_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative complex numbers
    x = tf.constant([-1 - 1j, -2 - 2j, -3 - 3j]).numpy()
    name = "negative_complex_conj_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Zero complex numbers
    x = tf.constant([0 + 0j, 0 + 0j, 0 + 0j]).numpy()
    name = "zero_complex_conj_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different complex numbers
    x = tf.constant([1 + 2j, 3 - 4j, -5 + 6j]).numpy()
    name = "complex_conj_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex numbers
    x = tf.constant([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]).numpy()
    name = "complex_conj_3"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D Real numbers
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]]).numpy()
    name = "real_conj_2"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Complex64
    x = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    name = "complex64_conj_1"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 10: 3D complex numbers
    x = tf.constant([[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],[[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j]]]).numpy()
    name = "complex_conj_4"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.conj"] = tf_math_conj_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.conj'.")

check_valid('tf.math.conj', generated_inputs['tf.math.conj'], lib="tf", suffix=0)
