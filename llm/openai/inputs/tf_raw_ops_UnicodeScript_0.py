
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_unicode_script_inputs():
    list_of_inputs = []

    # Input 1: 1D small integers
    name = "basic_1d"
    input_arr = np.array([1, 31, 38], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 2: 2D ASCII-like values
    name = "ascii_2d"
    input_arr = np.array([[72, 101, 108, 108, 111],
                          [87, 111, 114, 108, 100]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 3: Negative and zero values
    name = "negatives_and_zero"
    input_arr = np.array([-1, -100, 0, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 4: Empty 1D array
    name = "empty_1d"
    input_arr = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 5: Scalar (0-D) tensor
    name = "scalar_65"
    input_arr = np.array(65, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 6: 3D array with mixed valid and invalid code points
    name = "mixed_3d"
    input_arr = np.array([
        [[0x10FFFF, 0x110000, 0x0041],
         [0xAC00,    0x3042,   0x30A2]],
        [[0x4E00,    0x09FF,   0x3400],
         [0xD800,    0xDBFF,   0xDC00]]
    ], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 7: Boundary values and common scripts
    name = "boundaries_and_common"
    input_arr = np.array([0, 0x10FFFF, 0x007A, 0x0416], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 8: Devanagari-related code points
    name = "devanagari_2x2"
    input_arr = np.array([[0x0905, 0x0939],
                          [0x0966, 0x096F]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 9: Surrogate range and adjacent
    name = "surrogates_and_pua"
    input_arr = np.array([0xD800, 0xDFFF, 0xE000], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 10: 4D array with CJK characters
    name = "cjk_4d"
    input_arr = np.array([[[[28450, 23383, 20013]],
                           [[22283, 20108, 24180]]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 11: Arange reshaped to 3x3
    name = "arange_3x3"
    input_arr = np.arange(9, dtype=np.int32).reshape(3, 3)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    # Input 12: Extreme int32 values and boundary checks
    name = "extreme_int32_and_boundary"
    input_arr = np.array([-2147483648, 2147483647, 1114111, 1114112], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr}))

    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_raw_ops_unicode_script_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.UnicodeScript' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeScript'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.UnicodeScript', generated_inputs['tf.raw_ops.UnicodeScript'], lib="tf", suffix=0)
