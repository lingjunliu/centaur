
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_get_session_handle_inputs():
    list_of_inputs = []

    # Input 1
    value = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    input_dict = {"value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    value = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    input_dict = {"value": value, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    value = tf.constant(np.array([True, False, True], dtype=np.bool_))
    input_dict = {"value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    value = tf.constant(np.array(["hello", "world"], dtype=np.string_))
    input_dict = {"value": value, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    value = tf.constant(np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
    input_dict = {"value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    value = tf.constant(np.array([[-1, -2], [-3, -4]], dtype=np.int64))
    input_dict = {"value": value, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    value = tf.constant(np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64))
    input_dict = {"value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor
    value = tf.constant(np.random.rand(2, 3, 4).astype(np.float32))
    input_dict = {"value": value, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensor
    value = tf.constant(np.array([], dtype=np.int32))
    input_dict = {"value": value, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with different shape
    value = tf.constant(np.random.rand(5, 5).astype(np.float32))
    input_dict = {"value": value, "name": "test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.GetSessionHandle"] = tf_raw_ops_get_session_handle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.GetSessionHandle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.GetSessionHandle'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.GetSessionHandle', generated_inputs['tf.raw_ops.GetSessionHandle'], lib="tf", suffix=0)
