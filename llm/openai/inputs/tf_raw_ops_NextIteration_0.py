
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_NextIteration_inputs():
    list_of_inputs = []

    data = np.array(5, dtype=np.int32)
    input_dict = {"name": "next_iter_case_1", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1.0, -2.5, np.nan, np.inf], dtype=np.float32)
    input_dict = {"name": "next_iter_case_2", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.int64)
    input_dict = {"name": "next_iter_case_3", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[[1.5, -0.5], [2.25, -3.75]]], dtype=np.float16)
    input_dict = {"name": "next_iter_case_4", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[True, False, True], [False, True, False]], dtype=bool)
    input_dict = {"name": "next_iter_case_5", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([1+2j, -3-4j, 0+0j], dtype=np.complex64)
    input_dict = {"name": "next_iter_case_6", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([b"hello", b"world", b"tf"], dtype=np.object_)
    input_dict = {"name": "next_iter_case_7", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([[0, 255, 128], [64, 32, 16], [200, 100, 50]], dtype=np.uint8)
    input_dict = {"name": "next_iter_case_8", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([], dtype=np.float32)
    input_dict = {"name": "next_iter_case_9", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.random.RandomState(0).randn(2, 1, 3, 4).astype(np.float32)
    input_dict = {"name": "next_iter_case_10", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.empty((2, 0), dtype=np.int32)
    input_dict = {"name": "next_iter_case_11", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    data = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    input_dict = {"name": "next_iter_case_12", "data": data}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.NextIteration"] = tf_raw_ops_NextIteration_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.NextIteration' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.NextIteration'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.NextIteration', generated_inputs['tf.raw_ops.NextIteration'], lib="tf", suffix=0)
