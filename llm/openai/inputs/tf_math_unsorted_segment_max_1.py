
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [4, 3, 2, 1]], dtype=np.int32)
    segment_ids = np.array([0, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "case1"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(3*2*4).reshape(3, 2, 4).astype(np.float32) - 5.0)
    segment_ids = np.array([0, 1, 0], dtype=np.int64)
    num_segments = 3
    name = "case2"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([10, -2, 7, 0, 5], dtype=np.int64)
    segment_ids = np.array([0, 1, -1, 1, 0], dtype=np.int32)
    num_segments = 2
    name = "case3"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(2*3*2*2, dtype=np.float64).reshape(2, 3, 2, 2) * 0.5
    segment_ids = np.array([[0, 1, 2],
                            [3, 4, 0]], dtype=np.int64)
    num_segments = 5
    name = "case4"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([255, 0, 128, 64, 200, 1], dtype=np.uint8)
    segment_ids = np.array([0, 1, 1, 0, 1, 0], dtype=np.int32)
    num_segments = 3
    name = "case5"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(20).reshape(4, 5).astype(np.float16) - np.float16(10))
    segment_ids = np.array([0, -1, 1, 1], dtype=np.int64)
    num_segments = 3
    name = "case6"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = (np.arange(3*4*2) % 50).astype(np.int16).reshape(3, 4, 2)
    segment_ids = np.array([[0, 1, 2, 3],
                            [4, 5, 0, 1],
                            [2, 3, 4, 5]], dtype=np.int32)
    num_segments = 6
    name = "case7"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1, 2, -3],
                     [4, 5, -6],
                     [7, 8, 9],
                     [-10, 11, 12]], dtype=np.int8).reshape(2, 2, 3)
    segment_ids = np.array([[0, 1],
                            [2, 0]], dtype=np.int64)
    num_segments = 3
    name = "case8"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(5, dtype=np.int32).reshape(5, 1)
    segment_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    num_segments = 6
    name = "case9"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(2*3*1*2, dtype=np.int64).reshape(2, 3, 1, 2)
    segment_ids = np.array([[[0], [1], [2]],
                            [[3], [0], [1]]], dtype=np.int32)
    num_segments = 4
    name = "case10"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1e9, -2e9],
                     [3.5, -4.2],
                     [0.0, -7.7]], dtype=np.float64)
    segment_ids = np.array([2, 0, 2], dtype=np.int32)
    num_segments = 6
    name = "case11"
    input_dict = {"data": data, "segment_ids": segment_ids, "num_segments": num_segments, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max_1"] = tf_math_unsorted_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max_1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max_1'], lib="tf", suffix=1)
