
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SegmentSum_inputs():
    list_of_inputs = []

    data = np.array([[1.0, 2.0, 3.0, 4.0],
                     [4.0, 3.0, 2.0, 1.0],
                     [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    segment_ids = np.array([0, 0, 1], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_1", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([-3, 0, 2, -1, 5, -2], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_2", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    segment_ids = np.array([0, 0, 1, 2], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_3", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
    segment_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_4", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-10, 20, -30],
                     [40, -50, 60],
                     [70, -80, 90],
                     [-100, 110, -120],
                     [130, -140, 150],
                     [160, -170, 180]], dtype=np.int16)
    segment_ids = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_5", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1, -2], [3, -4]],
                     [[5, 6], [-7, 8]],
                     [[-9, 10], [11, -12]],
                     [[13, -14], [15, 16]]], dtype=np.int8)
    segment_ids = np.array([0, 2, 2, 2], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_6", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[1+2j, -3+4j, 5-6j],
                     [7+0j, -1-1j, 2+2j],
                     [0+3j, -4-5j, 6+6j]], dtype=np.complex64)
    segment_ids = np.array([0, 1, 1], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_7", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16],
                     [17, 18, 19, 20]], dtype=np.int64)
    segment_ids = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_8", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1, 2], [3, 4]],
                     [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=np.float16)
    segment_ids = np.array([0, 0, 2], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_9", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1+0j, 0+1j, 3+3j, -2-2j], dtype=np.complex128)
    segment_ids = np.array([0, 1, 1, 3], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_10", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.arange(35, dtype=np.float32).reshape(7, 5)
    segment_ids = np.array([0, 0, 1, 1, 1, 2, 3], dtype=np.int32)
    input_dict = {"name": "seg_sum_case_11", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[100, -50, 25],
                     [0, 0, 0],
                     [1, 2, 3],
                     [9, 9, 9]], dtype=np.int32)
    segment_ids = np.array([0, 0, 0, 5], dtype=np.int64)
    input_dict = {"name": "seg_sum_case_12", "data": data, "segment_ids": segment_ids}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.SegmentSum"] = tf_raw_ops_SegmentSum_inputs()

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
