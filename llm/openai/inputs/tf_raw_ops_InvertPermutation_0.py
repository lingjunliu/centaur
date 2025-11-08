
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_invert_permutation_inputs():
    list_of_inputs = []

    x = np.array([0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "single_element_int32", "x": x}))

    x = np.array([1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "two_elements_swap_int64", "x": x}))

    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "identity_len5_int32", "x": x}))

    x = np.array([4, 3, 2, 1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "reverse_len5_int64", "x": x}))

    x = np.array([3, 0, 6, 1, 4, 2, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "random_len7_int32", "x": x}))

    x = np.array([7, 2, 9, 0, 5, 1, 8, 6, 4, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "random_len10_int64", "x": x}))

    x = np.array([2, 8, 4, 0, 6, 1, 3, 5, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "mixed_pattern_len9_int32", "x": x}))

    x = np.array([2, 0, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "cycle_len3_int64", "x": x}))

    x = np.array([1, 5, 3, 2, 4, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "custom_len6_int32", "x": x}))

    x = np.array([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "reverse_len12_int64", "x": x}))

    x = np.array([1, 2, 3, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": "rotate_left_len4_int32", "x": x}))

    x = np.array([(i + 3) % 16 for i in range(16)], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"name": "cyclic_shift_len16_int64", "x": x}))

    return list_of_inputs

generated_inputs["tf.raw_ops.InvertPermutation"] = tf_raw_ops_invert_permutation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.InvertPermutation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.InvertPermutation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.InvertPermutation', generated_inputs['tf.raw_ops.InvertPermutation'], lib="tf", suffix=0)
