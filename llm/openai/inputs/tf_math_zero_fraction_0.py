
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_zero_fraction_inputs():
    list_of_inputs = []

    value = np.array([0.0, 1.0, -2.5, 0.0], dtype=np.float32)
    name = "basic_float_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([[0, -1, 2], [3, 0, 0]], dtype=np.int32)
    name = "int32_matrix"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.zeros((2, 3, 4), dtype=np.float16)
    value[0, 1, 2] = -1.5
    value[1, 2, 3] = 0.5
    name = "float16_3d_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array(0, dtype=np.int64)
    name = "scalar_zero_int64"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array(3.14159, dtype=np.float64)
    name = "scalar_nonzero_float64"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([], dtype=np.float32)
    name = "empty_vector_float32"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.zeros((2, 2, 2, 3), dtype=np.int16)
    value[0, 0, 0, 0] = 10
    value[1, 1, 1, 2] = -5
    name = "int16_4d_tensor"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([0+0j, 1+0j, 0+2j, 3+4j, 0+0j], dtype=np.complex64)
    name = "complex64_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([[0+0j, 0+0j], [5+0j, 0+1j]], dtype=np.complex128)
    name = "complex128_matrix"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([0, 255, 0, 128, 1, 0], dtype=np.uint8)
    name = "uint8_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([-128, 0, 127, 0, -1], dtype=np.int8)
    name = "int8_vector"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.array([0.0, np.nan, np.inf, -np.inf, 1.0, 0.0], dtype=np.float32)
    name = "float_with_nan_inf"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    value = np.zeros((0, 5), dtype=np.int64)
    name = "empty_2d_int64"
    input_dict = {"value": value, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.zero_fraction"] = tf_math_zero_fraction_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.zero_fraction' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.zero_fraction'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.zero_fraction', generated_inputs['tf.math.zero_fraction'], lib="tf", suffix=0)
