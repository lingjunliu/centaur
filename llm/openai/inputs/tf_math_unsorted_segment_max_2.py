
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_unsorted_segment_max_2_inputs():
    list_of_inputs = []

    # Input 1
    data = np.array([1, 3, -2, 5], dtype=np.int32)
    segment_ids = np.array([0, 1, 0, -1], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "case1_basic_int32_with_negative_id"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    data = np.array([[1.0, 2.5, -3.0, 4.5],
                     [5.5, -6.0, 7.2, 8.8],
                     [4.0, 3.3, 2.2, 1.1]], dtype=np.float32)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    name = "rows_prefix_float32"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    data = np.array([[1, 200, 3],
                     [4, 5, 6]], dtype=np.uint8)
    segment_ids = np.array([[0, 1, -1],
                            [1, 0, 1]], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    name = "full_shape_ids_uint8_with_negative"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    data = np.array([[[1.0, -2.0, 3.0, 4.0],
                      [5.0, 6.0, -7.0, 8.0],
                      [9.0, -10.0, 11.0, 12.0]],
                     [[-1.0, 2.0, -3.0, -4.0],
                      [-5.0, -6.0, 7.0, -8.0],
                      [-9.0, 10.0, -11.0, -12.0]]], dtype=np.float64)
    segment_ids = np.array([1, 0], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int64)
    name = "3d_firstdim_float64"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    data = np.array([[[10, -20],
                      [30, -40]],
                     [[-50, 60],
                      [70, -80]],
                     [[90, -100],
                      [110, -120]]], dtype=np.int16)
    segment_ids = np.array([[0, 1],
                            [1, 2],
                            [2, 0]], dtype=np.int32)
    num_segments = np.array(3, dtype=np.int32)
    name = "2d_prefix_int16"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    data = np.array([[[-1, 2],
                      [3, -4]],
                     [[5, -6],
                      [-7, 8]]], dtype=np.int8)
    segment_ids = np.array([[[0, 1],
                             [1, 0]],
                            [[1, 0],
                             [0, 1]]], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    name = "elemwise_ids_int8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    data = np.array([65530, 1, 500, 60000, 42, 0], dtype=np.int32)
    segment_ids = np.array([2, 2, 7, 2, 7, 7], dtype=np.int32)
    num_segments = np.array(10, dtype=np.int32)
    name = "int32_sparse_segments_with_gaps"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    data = np.array([[[[1.0, -2.0],
                       [3.0, -4.0]],
                      [[-5.0, 6.0],
                       [-7.0, 8.0]]],
                     [[[9.0, -10.0],
                       [11.0, -12.0]],
                      [[-13.0, 14.0],
                       [-15.0, 16.0]]]], dtype=np.float16)
    segment_ids = np.array([0, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int32)
    name = "float16_4d_firstdim_prefix"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    data = np.array([-10, 0, 10], dtype=np.int64)
    segment_ids = np.array([-1, -1, -1], dtype=np.int64)
    num_segments = np.array(3, dtype=np.int64)
    name = "all_dropped_int64"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    data = np.array([[0.1, -0.2, 0.3],
                     [1.5, -1.0, 2.0],
                     [-0.5, 0.7, -0.9],
                     [3.2, -3.3, 3.4]], dtype=np.float64)
    segment_ids = np.array([0, 2, 1, 2], dtype=np.int32)
    num_segments = np.array(4, dtype=np.int32)
    name = "float64_2d_rows_prefix"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    data = np.array([[[4000000000, 123],
                      [987654321, 2222222222],
                      [3333333333, 4294967295]],
                     [[111111111, 222222222],
                      [333333333, 444444444],
                      [555555555, 666666666]]], dtype=np.int64)
    segment_ids = np.array([[0, 4, 1],
                            [2, 4, 0]], dtype=np.int32)
    num_segments = np.array(5, dtype=np.int64)
    name = "int64_3d_2d_prefix_with_gaps"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    data = np.array([[2**62], [2**40], [2**32]], dtype=np.int64)
    segment_ids = np.array([[0], [0], [1]], dtype=np.int64)
    num_segments = np.array(3, dtype=np.int32)
    name = "int64_2d_fullshape_ids"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max_2"] = tf_math_unsorted_segment_max_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max_2'], lib="tf", suffix=2)
