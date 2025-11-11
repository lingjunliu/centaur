
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_unsorted_segment_max_inputs():
    list_of_inputs = []

    data1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=np.int32)
    segment_ids1 = np.array([0, 1, 0], dtype=np.int32)
    num_segments1 = 2
    name1 = "test1"
    input_dict1 = {
        "data": data1,
        "segment_ids": segment_ids1,
        "num_segments": num_segments1,
        "name": name1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    data2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    segment_ids2 = np.array([0, 1, 0], dtype=np.int32)
    num_segments2 = 2
    name2 = "test2"
    input_dict2 = {
        "data": data2,
        "segment_ids": segment_ids2,
        "num_segments": num_segments2,
        "name": name2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    data3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    segment_ids3 = np.array([0, 1, 0], dtype=np.int64)
    num_segments3 = 2
    name3 = "test3"
    input_dict3 = {
        "data": data3,
        "segment_ids": segment_ids3,
        "num_segments": num_segments3,
        "name": name3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    data4 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    segment_ids4 = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    num_segments4 = 2
    name4 = "test4"
    input_dict4 = {
        "data": data4,
        "segment_ids": segment_ids4,
        "num_segments": num_segments4,
        "name": name4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    data5 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    segment_ids5 = np.array([0, 1], dtype=np.int32)
    num_segments5 = 2
    name5 = "test5"
    input_dict5 = {
        "data": data5,
        "segment_ids": segment_ids5,
        "num_segments": num_segments5,
        "name": name5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    data6 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    segment_ids6 = np.array([0, 1], dtype=np.int32)
    num_segments6 = 2
    name6 = "test6"
    input_dict6 = {
        "data": data6,
        "segment_ids": segment_ids6,
        "num_segments": num_segments6,
        "name": name6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["tf.math.unsorted_segment_max"] = tf_math_unsorted_segment_max_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.unsorted_segment_max' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.unsorted_segment_max'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.unsorted_segment_max', generated_inputs['tf.math.unsorted_segment_max'], lib="tf", suffix=0)
