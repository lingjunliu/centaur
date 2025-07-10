
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_angle_inputs():
    list_of_inputs = []

    # Input 1: Complex64 tensor
    input1 = tf.constant([1 + 1j, 2 + 2j, 3 + 3j], dtype=tf.complex64)
    input_dict = {"input": input1, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex128 tensor
    input2 = tf.constant([-1 - 1j, -2 - 2j, -3 - 3j], dtype=tf.complex128)
    input_dict = {"input": input2, "name": "complex128_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 tensor
    input3 = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
    input_dict = {"input": input3, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 tensor
    input4 = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float64)
    input_dict = {"input": input4, "name": "float64_input"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional Complex64 tensor
    input5 = tf.constant([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=tf.complex64)
    input_dict = {"input": input5, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional Float64 tensor
    input6 = tf.constant([[-1.0, 0.0], [1.0, 2.0]], dtype=tf.float64)
    input_dict = {"input": input6, "name": "multi_float64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64 with zero real part
    input7 = tf.constant([1j, 2j, 3j], dtype=tf.complex64)
    input_dict = {"input": input7, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 with zero imaginary part
    input8 = tf.constant([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=tf.complex128)
    input_dict = {"input": input8, "name": "zero_imag"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32 with negative values
    input9 = tf.constant([-5.0, -2.5, -1.0], dtype=tf.float32)
    input_dict = {"input": input9, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex128 with mixed signs
    input10 = tf.constant([-1 + 1j, 2 - 2j, -3 - 3j], dtype=tf.complex128)
    input_dict = {"input": input10, "name": "mixed_signs"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.angle"] = tf_math_angle_inputs()
for i in range(len(generated_inputs["tf.math.angle"])):
    generated_inputs["tf.math.angle"][i]["input"] = generated_inputs["tf.math.angle"][i]["input"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.angle'.")

check_valid('tf.math.angle', generated_inputs['tf.math.angle'], lib="tf", suffix=0)
