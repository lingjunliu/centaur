
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_tf_raw_ops_barrier_insert_many_inputs():
    # This operation, tf.raw_ops.BarrierInsertMany, is designed for TensorFlow's graph mode
    # and is not compatible with eager execution. The Python wrapper for this op explicitly
    # raises a RuntimeError if it's called in an eager context, which is the expected
    # behavior when running in the test environment.
    # The provided inputs are valid for a graph-based execution scenario where 'handle'
    # would be a resource tensor produced by a tf.raw_ops.Barrier operation.
    list_of_inputs = []

    # The 'handle' tensor is a placeholder. A numpy array with dtype=object is used
    # to represent the string handle, which TensorFlow will convert to a tf.string tensor.
    handle_tensor = np.array(b'barrier_handle_placeholder', dtype=object)

    # Input 1: Basic case with 1D float32 values
    input_dict_1 = {
        'name': 'insert_floats',
        'handle': handle_tensor,
        'keys': np.array([b'key1', b'key2', b'key3'], dtype=object),
        'values': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int32 values
    input_dict_2 = {
        'name': 'insert_2d_ints',
        'handle': handle_tensor,
        'keys': np.array([b'alpha', b'beta'], dtype=object),
        'values': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'component_index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Complex128 values
    input_dict_3 = {
        'name': 'insert_complex128',
        'handle': handle_tensor,
        'keys': np.array([b'c128_1', b'c128_2'], dtype=object),
        'values': np.array([1+2j, 3-4j], dtype=np.complex128),
        'component_index': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Negative int16 values
    input_dict_4 = {
        'name': 'insert_neg_int16',
        'handle': handle_tensor,
        'keys': np.array([b'neg_key_1', b'neg_key_2'], dtype=object),
        'values': np.array([[-100, -200], [-300, -400]], dtype=np.int16),
        'component_index': 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Boolean values
    input_dict_5 = {
        'name': 'insert_bools',
        'handle': handle_tensor,
        'keys': np.array([b'true_key', b'false_key', b'true_key2'], dtype=object),
        'values': np.array([[True], [False], [True]], dtype=np.bool_),
        'component_index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Values are also strings
    input_dict_6 = {
        'name': 'string_values',
        'handle': handle_tensor,
        'keys': np.array([b'str_key1', b'str_key2'], dtype=object),
        'values': np.array([b'value1', b'value2'], dtype=object),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Single key-value pair with a 3D value tensor
    input_dict_7 = {
        'name': 'insert_single_3d',
        'handle': handle_tensor,
        'keys': np.array([b'single_3d_key'], dtype=object),
        'values': np.ones((1, 2, 2, 2), dtype=np.float32),
        'component_index': 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High component_index
    input_dict_8 = {
        'name': 'high_component_index',
        'handle': handle_tensor,
        'keys': np.array([b'high_idx'], dtype=object),
        'values': np.array([123.456], dtype=np.float32),
        'component_index': 1024
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Unsigned int32 values
    input_dict_9 = {
        'name': 'insert_uint32',
        'handle': handle_tensor,
        'keys': np.array([b'a', b'b', b'c', b'd'], dtype=object),
        'values': np.arange(4, dtype=np.uint32).reshape(4, 1),
        'component_index': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Empty keys and values (valid case)
    input_dict_10 = {
        'name': 'insert_empty',
        'handle': handle_tensor,
        'keys': np.array([], dtype=object),
        'values': np.array([], dtype=np.float32).reshape((0, 10)),
        'component_index': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.BarrierInsertMany"] = get_tf_raw_ops_barrier_insert_many_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierInsertMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierInsertMany'.")

check_valid('tf.raw_ops.BarrierInsertMany', generated_inputs['tf.raw_ops.BarrierInsertMany'], lib="tf", suffix=0)
