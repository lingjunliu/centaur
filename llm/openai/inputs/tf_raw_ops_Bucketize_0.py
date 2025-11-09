
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_Bucketize_inputs():
    list_of_inputs = []

    # Input 1
    name = "basic_int32_2d"
    input_arr = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.int32)
    boundaries = [0.0, 10.0, 100.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 2
    name = "float32_1d_neg_pos"
    input_arr = np.array([-2.5, 0.5, 1.1, 3.0, -1.0], dtype=np.float32)
    boundaries = [-1.0, 0.5, 2.5]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 3
    name = "int64_3d_random"
    input_arr = np.random.randint(-200, 1200, size=(2, 2, 3)).astype(np.int64)
    boundaries = [-100.0, 0.0, 100.0, 1000.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 4
    name = "scalar_float64_single_boundary"
    input_arr = np.array(3.14, dtype=np.float64)
    boundaries = [0.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 5
    name = "empty_boundaries_2d"
    input_arr = np.array([[1.2, -3.4], [5.6, 0.0]], dtype=np.float32)
    boundaries = []
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 6
    name = "int32_4d"
    input_arr = np.arange(12, dtype=np.int32).reshape(2, 1, 3, 2)
    boundaries = [-10.0, 10.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 7
    name = "int64_mixed_2d"
    input_arr = np.array([[-10, -3, 0], [2, 4, 9], [11, -6, 3]], dtype=np.int64)
    boundaries = [-5.5, -2.0, 3.3]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 8
    name = "empty_input_1d"
    input_arr = np.array([], dtype=np.float32)
    boundaries = [0.0, 1.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 9
    name = "large_ints_2d"
    input_arr = np.array([[-2000000000, 0], [2000000000, -1]], dtype=np.int32)
    boundaries = [-1000000000.0, 0.0, 1000000000.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 10
    name = "float64_5d_small_vals"
    input_arr = np.random.randn(1, 2, 1, 2, 2).astype(np.float64)
    boundaries = [-0.1, 0.1]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 11
    name = "int32_many_boundaries_1d"
    input_arr = np.arange(-12, 13, 3, dtype=np.int32)
    boundaries = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    # Input 12
    name = "float32_decimals_2d"
    input_arr = np.array([[-4.2, -1.2, -0.5, 2.7, 10.1],
                          [0.0, 1.5, -3.5, 9.9, 2.6]], dtype=np.float32)
    boundaries = [-3.5, -1.2, 0.0, 2.7, 9.9]
    list_of_inputs.append(copy.deepcopy({"name": name, "input": input_arr, "boundaries": boundaries}))

    return list_of_inputs

generated_inputs["tf.raw_ops.Bucketize"] = tf_raw_ops_Bucketize_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bucketize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bucketize', generated_inputs['tf.raw_ops.Bucketize'], lib="tf", suffix=0)
