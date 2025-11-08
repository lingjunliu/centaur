
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_placeholder_with_default_inputs():
    list_of_inputs = []

    # Input 1
    arr = np.array([1.0, -2.5, 3.3], dtype=np.float32)
    shape = [-1]
    input_dict = {"name": "case1_f32_vec_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shape = [2, 3]
    input_dict = {"name": "case2_i32_mat", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arr = np.array(True, dtype=bool)
    shape = []
    input_dict = {"name": "case3_bool_scalar", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arr = np.empty((2, 0, 4), dtype=np.complex64)
    shape = [2, 0, 4]
    input_dict = {"name": "case4_c64_empty_axis", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arr = (np.arange(1 * 2 * 3 * 4, dtype=np.float64).reshape(1, 2, 3, 4) * 0.5) - 10.0
    shape = [-1, 2, -1, 4]
    input_dict = {"name": "case5_f64_4d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arr = np.array([], dtype=np.int64)
    shape = [0]
    input_dict = {"name": "case6_i64_empty1d", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    arr = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    shape = [2, 2]
    input_dict = {"name": "case7_u8_mat", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    arr = np.array(
        [
            [[-1.0, 2.0, -3.5], [4.2, 0.0, -0.1], [7.7, -8.8, 9.9]],
            [[1.1, -2.2, 3.3], [-4.4, 5.5, -6.6], [7.7, -8.8, 9.9]],
            [[0.0, 0.0, 0.0], [1.5, -1.5, 1.5], [-1.5, 1.5, -1.5]],
        ],
        dtype=np.float16,
    )
    shape = [3, 3, 3]
    input_dict = {"name": "case8_f16_3d", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arr = np.ones((2, 3, 4, 5, 6), dtype=np.int16) * -7
    shape = [-1, -1, 4, 5, 6]
    input_dict = {"name": "case9_i16_5d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    arr = np.array(42, dtype=np.int8)
    shape = []
    input_dict = {"name": "case10_i8_scalar", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    arr = np.array(
        [
            [True, False, True],
            [False, False, True],
            [True, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )
    shape = [-1, 3]
    input_dict = {"name": "case11_bool_2d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    shape = [3, -1]
    input_dict = {"name": "case12_f32_2d_partial", "input": arr, "shape": shape}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.PlaceholderWithDefault"] = tf_raw_ops_placeholder_with_default_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PlaceholderWithDefault' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PlaceholderWithDefault'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.PlaceholderWithDefault', generated_inputs['tf.raw_ops.PlaceholderWithDefault'], lib="tf", suffix=0)
