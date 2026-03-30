
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_GetSessionHandleV2_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensor
    value = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"value": value, "name": "handle1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor
    value = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {"value": value, "name": "handle2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer tensor
    value = np.array([[1, 2], [3, 4]], dtype=np.int64)
    input_dict = {"value": value, "name": "handle3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean tensor
    value = np.array([True, False, True], dtype=np.bool_)
    input_dict = {"value": value, "name": "handle4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String tensor (bytes)
    value = np.array([b"hello", b"world"])
    input_dict = {"value": value, "name": "handle5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    value = np.array([], dtype=np.int32)
    input_dict = {"value": value, "name": "handle6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float tensor
    value = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict = {"value": value, "name": "handle7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer values
    value = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {"value": value, "name": "handle8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer values
    value = np.array([2**31-1, -(2**31)], dtype=np.int64)
    input_dict = {"value": value, "name": "handle9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with a different shape
    value = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]], dtype=np.int32)
    input_dict = {"value": value, "name": "handle10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GetSessionHandleV2"] = tf_raw_ops_GetSessionHandleV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandleV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandleV2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.GetSessionHandleV2', generated_inputs['tf.raw_ops.GetSessionHandleV2'], lib="tf", suffix=0)
