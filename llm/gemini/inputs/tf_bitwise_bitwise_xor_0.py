
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_bitwise_bitwise_xor_inputs():
    list_of_inputs = []
    
    # Input 1: 1D arrays, int32
    list_of_inputs.append({
        "x": np.array([0, 5, 3, 14], dtype=np.int32),
        "y": np.array([5, 0, 7, 11], dtype=np.int32),
        "name": "xor_int32"
    })
    
    # Input 2: 2D arrays, int32 with negative values
    list_of_inputs.append({
        "x": np.array([[-1, 2], [-3, 4]], dtype=np.int32),
        "y": np.array([[5, -6], [7, -8]], dtype=np.int32),
        "name": "xor_int32_negative"
    })

    # Input 3: 1D arrays, int32 positive
    list_of_inputs.append({
        "x": np.array([255, 0, 127], dtype=np.int32),
        "y": np.array([0, 255, 128], dtype=np.int32),
        "name": "xor_int32_positive"
    })

    # Input 4: Scalar values (0D arrays), int64
    list_of_inputs.append({
        "x": np.array(922337203685477580, dtype=np.int64),
        "y": np.array(-922337203685477580, dtype=np.int64),
        "name": "xor_scalar_int64"
    })

    # Input 5: 3D arrays, int32
    list_of_inputs.append({
        "x": np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32),
        "y": np.array([[[-1, 2], [-3, 4]], [[-5, 6], [-7, 8]]], dtype=np.int32),
        "name": "xor_int32_3d"
    })

    # Input 6: 4D arrays, int32
    list_of_inputs.append({
        "x": np.ones((2, 2, 2, 2), dtype=np.int32) * 65535,
        "y": np.zeros((2, 2, 2, 2), dtype=np.int32),
        "name": "xor_int32_4d"
    })

    # Input 7: 2D arrays, int64
    list_of_inputs.append({
        "x": np.array([[123456, 789012], [345678, 901234]], dtype=np.int64),
        "y": np.array([[654321, 210987], [876543, 432109]], dtype=np.int64),
        "name": "xor_int64_2d"
    })

    # Input 8: Broadcasting 2D and 1D arrays, int32
    list_of_inputs.append({
        "x": np.array([[1], [2], [3]], dtype=np.int32),
        "y": np.array([4, 5, 6], dtype=np.int32),
        "name": "xor_broadcast"
    })

    # Input 9: Large values in int64
    list_of_inputs.append({
        "x": np.array([4611686018427387903, 0], dtype=np.int64),
        "y": np.array([0, 4611686018427387903], dtype=np.int64),
        "name": "xor_int64_large"
    })

    # Input 10: 1D arrays of int64 with negative values
    list_of_inputs.append({
        "x": np.array([-100, 200, -300], dtype=np.int64),
        "y": np.array([400, -500, 600], dtype=np.int64),
        "name": "xor_int64_1d"
    })

    return list_of_inputs

generated_inputs["tf.bitwise.bitwise_xor"] = tf_bitwise_bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.bitwise.bitwise_xor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.bitwise.bitwise_xor'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.bitwise.bitwise_xor', generated_inputs['tf.bitwise.bitwise_xor'], lib="tf", suffix=0)
