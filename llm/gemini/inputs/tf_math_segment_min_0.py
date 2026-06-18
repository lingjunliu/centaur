
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_segment_min_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 1.5, 2.5]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 2: 1D int32 array, with name
    data = np.array([10, -5, 20, 30, -100], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    name = "segment_min_1d"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 3: 3D float64 array
    data = np.random.randn(4, 2, 2).astype(np.float64)
    segment_ids = np.array([0, 1, 1, 2], dtype=np.int64)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 4: 2D uint8 array
    data = np.array([[10, 20], [30, 40], [5, 15], [25, 35]], dtype=np.uint8)
    segment_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    name = "uint8_test"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 5: 2D int16 array with negative values
    data = np.array([[-10, 20], [-30, -40], [50, -15], [25, -35], [0, 0]], dtype=np.int16)
    segment_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 6: 1D float32 array with negative values
    data = np.array([-1.5, -2.5, 3.0, 4.2, -5.5], dtype=np.float32)
    segment_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    name = "neg_float"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 7: 4D float32 array
    data = np.random.randn(3, 2, 2, 2).astype(np.float32)
    segment_ids = np.array([0, 0, 0], dtype=np.int64)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 8: 2D int64 array with gaps in segment_ids
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    segment_ids = np.array([1, 1, 3, 3], dtype=np.int32)
    name = "gaps_in_ids"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 9: 2D int8 array
    data = np.array([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]], dtype=np.int8)
    segment_ids = np.array([0, 1], dtype=np.int32)
    name = None
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    # Input 10: 3D float64 array, single elements in segments
    data = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]], dtype=np.float64)
    segment_ids = np.array([0, 1, 2], dtype=np.int32)
    name = "single_element_segments"
    list_of_inputs.append({"data": data, "segment_ids": segment_ids, "name": name})

    return list_of_inputs

generated_inputs["tf.math.segment_min"] = tf_math_segment_min_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.segment_min' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.segment_min'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.segment_min', generated_inputs['tf.math.segment_min'], lib="tf", suffix=0)
