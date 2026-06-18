
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_segment_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D Float32 array, 4 elements, 2 segments
    list_of_inputs.append({
        "data": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "segment_ids": np.array([0, 0, 1, 1], dtype=np.int32),
        "name": "segment_prod_1d_float32"
    })

    # Input 2: 2D Int32 array, shape (3, 2), 2 segments
    list_of_inputs.append({
        "data": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "segment_ids": np.array([0, 0, 1], dtype=np.int32),
        "name": "segment_prod_2d_int32"
    })

    # Input 3: 3D Float64 array, shape (4, 2, 2), 3 segments
    list_of_inputs.append({
        "data": np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[0.5, 1.5], [2.5, 3.5]],
            [[2.0, 2.0], [2.0, 2.0]]
        ], dtype=np.float64),
        "segment_ids": np.array([0, 1, 1, 2], dtype=np.int64),
        "name": "segment_prod_3d_float64"
    })

    # Input 4: 1D Complex64 array, 3 elements, 1 segment
    list_of_inputs.append({
        "data": np.array([1+1j, 2-1j, 3+2j], dtype=np.complex64),
        "segment_ids": np.array([0, 0, 0], dtype=np.int32),
        "name": "segment_prod_1d_complex"
    })

    # Input 5: 2D Uint8 array, shape (2, 2), 2 segments (one element each)
    list_of_inputs.append({
        "data": np.array([[5, 10], [15, 20]], dtype=np.uint8),
        "segment_ids": np.array([0, 1], dtype=np.int32),
        "name": "segment_prod_2d_uint8"
    })

    # Input 6: 1D Int64 array with negative values, 5 elements, 2 segments
    list_of_inputs.append({
        "data": np.array([-1, -2, 3, 4, -5], dtype=np.int64),
        "segment_ids": np.array([0, 0, 1, 1, 1], dtype=np.int64),
        "name": "segment_prod_1d_int64_neg"
    })

    # Input 7: 4D Float32 array, shape (3, 2, 2, 2), 1 segment
    list_of_inputs.append({
        "data": np.ones((3, 2, 2, 2), dtype=np.float32) * 2.0,
        "segment_ids": np.array([0, 0, 0], dtype=np.int32),
        "name": "segment_prod_4d_float32"
    })

    # Input 8: 2D Float32 array with single element in dimension 0, 1 segment
    list_of_inputs.append({
        "data": np.array([[1.5, 2.5]], dtype=np.float32),
        "segment_ids": np.array([0], dtype=np.int32),
        "name": "segment_prod_single_element_dim0"
    })

    # Input 9: 1D Int16 array, 3 elements, 3 segments (each element in its own segment)
    list_of_inputs.append({
        "data": np.array([10, 20, 30], dtype=np.int16),
        "segment_ids": np.array([0, 1, 2], dtype=np.int32),
        "name": "segment_prod_1d_int16"
    })

    # Input 10: 3D Float32 array, shape (5, 1, 3), 3 segments
    list_of_inputs.append({
        "data": np.array([
            [[-1.0, 2.0, -3.0]],
            [[4.0, -5.0, 6.0]],
            [[-7.0, 8.0, -9.0]],
            [[1.0, 1.0, 1.0]],
            [[2.0, 3.0, 4.0]]
        ], dtype=np.float32),
        "segment_ids": np.array([0, 0, 1, 2, 2], dtype=np.int32),
        "name": "segment_prod_3d_with_gaps_float32"
    })

    return list_of_inputs

generated_inputs["tf.math.segment_prod"] = tf_math_segment_prod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.segment_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.segment_prod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.segment_prod', generated_inputs['tf.math.segment_prod'], lib="tf", suffix=0)
