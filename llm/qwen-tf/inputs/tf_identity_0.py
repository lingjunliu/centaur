
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_identity_inputs():
    list_of_inputs = []

    input_arr = np.array([0.78], dtype=np.float32)
    input_dict = {"input": input_arr, "name": "float32_vector"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(5, dtype=np.int32)
    input_dict = {"input": input_arr, "name": "int32_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[-1, 0, 2], [3, -4, 5]], dtype=np.int64)
    input_dict = {"input": input_arr, "name": "int64_matrix_with_negatives"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[[np.nan, np.inf], [-np.inf, -1.5]]], dtype=np.float64)
    input_dict = {"input": input_arr, "name": "float64_3d_with_nan_inf"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([], dtype=np.int32)
    input_dict = {"input": input_arr, "name": "empty_int32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[True, False, True], [False, False, True]], dtype=bool)
    input_dict = {"input": input_arr, "name": "bool_matrix"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    input_dict = {"input": input_arr, "name": "complex64_vector"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(2 * 3 * 1 * 4, dtype=np.float16).reshape(2, 3, 1, 4)
    input_dict = {"input": input_arr, "name": "float16_4d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array(True, dtype=bool)
    input_dict = {"input": input_arr, "name": "bool_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.empty((2, 0, 3), dtype=np.float32)
    input_dict = {"input": input_arr, "name": "empty_axis_float32_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    input_dict = {"input": input_arr, "name": "uint8_3d_image"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.identity"] = tf_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.identity', generated_inputs['tf.identity'], lib="tf", suffix=0)
