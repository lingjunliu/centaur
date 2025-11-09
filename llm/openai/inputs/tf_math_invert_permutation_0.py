
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_invert_permutation_inputs():
    list_of_inputs = []

    x = np.array([0], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len1_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 0], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len2_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([0, 1, 2], dtype=np.int32)
    input_dict = {"x": x, "name": "case_identity_len3_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2, 0, 1], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len3_shuffle_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([3, 4, 0, 2, 1], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len5_example_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len5_reverse_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len6_rotate_left_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([6, 0, 2, 4, 1, 5, 3], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len7_custom_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([7, 3, 0, 2, 5, 1, 6, 4], dtype=np.int64)
    input_dict = {"x": x, "name": "case_len8_random_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len9_reverse_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1], dtype=np.int32)
    input_dict = {"x": x, "name": "case_len10_rotate_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(20, dtype=np.int64)[::-1]
    input_dict = {"x": x, "name": "case_len20_reverse_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.invert_permutation"] = tf_math_invert_permutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.invert_permutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.invert_permutation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.invert_permutation', generated_inputs['tf.math.invert_permutation'], lib="tf", suffix=0)
