
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_segment_sum_inputs():
    list_of_inputs = []

    data1 = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]], dtype=np.int32)
    segment_ids1 = np.array([0, 0, 1], dtype=np.int32)
    input_dict1 = {'name': 'segment_sum_1', 'data': data1, 'segment_ids': segment_ids1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    data2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    segment_ids2 = np.array([0, 1, 2, 3], dtype=np.int32)
    input_dict2 = {'name': 'segment_sum_2', 'data': data2, 'segment_ids': segment_ids2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    data3 = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    segment_ids3 = np.array([0, 1, 0], dtype=np.int32)
    input_dict3 = {'name': 'segment_sum_3', 'data': data3, 'segment_ids': segment_ids3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    data4 = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    segment_ids4 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    input_dict4 = {'name': 'segment_sum_4', 'data': data4, 'segment_ids': segment_ids4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    data5 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    segment_ids5 = np.array([0, 1, 0], dtype=np.int32)
    input_dict5 = {'name': 'segment_sum_5', 'data': data5, 'segment_ids': segment_ids5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    data6 = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    segment_ids6 = np.array([0, 1], dtype=np.int32)
    input_dict6 = {'name': 'segment_sum_6', 'data': data6, 'segment_ids': segment_ids6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    data7 = np.array([1, 2, 3], dtype=np.int16)
    segment_ids7 = np.array([0, 0, 0], dtype=np.int32)
    input_dict7 = {'name': 'segment_sum_7', 'data': data7, 'segment_ids': segment_ids7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    data8 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    segment_ids8 = np.array([0, 1], dtype=np.int32)
    input_dict8 = {'name': 'segment_sum_8', 'data': data8, 'segment_ids': segment_ids8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    data9 = np.array([[1, 2], [3, 4]], dtype=np.float16)
    segment_ids9 = np.array([0, 1], dtype=np.int32)
    input_dict9 = {'name': 'segment_sum_9', 'data': data9, 'segment_ids': segment_ids9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    data10 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    segment_ids10 = np.array([0, 1], dtype=np.int64)
    input_dict10 = {'name': 'segment_sum_10', 'data': data10, 'segment_ids': segment_ids10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.raw_ops.SegmentSum"] = tf_raw_ops_segment_sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.SegmentSum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SegmentSum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.SegmentSum', generated_inputs['tf.raw_ops.SegmentSum'], lib="tf", suffix=0)
