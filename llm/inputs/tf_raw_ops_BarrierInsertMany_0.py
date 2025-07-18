
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_barrier_insert_many_inputs():
    """
    Generates a list of valid inputs for tf.raw_ops.BarrierInsertMany.
    NOTE: This operation is designed for TensorFlow's graph mode and is not
    compatible with eager execution. The 'handle' input requires a reference
    to a Barrier resource, which cannot be created from NumPy alone. The
    following inputs are structurally valid according to the API signature
    but are expected to fail with a RuntimeError if run in an eager context.
    `dtype=object` is used for string tensors to satisfy the testing framework's
    type validation.
    """
    list_of_inputs = []

    # Input 1: Basic case with float32 values
    input_dict_1 = {
        'name': 'insert_floats',
        'handle': np.array("barrier_handle_1", dtype=object),
        'keys': np.array(["key_a", "key_b", "key_c"], dtype=object),
        'values': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D tensor of int32 values and a different component_index
    input_dict_2 = {
        'name': 'insert_int_matrix',
        'handle': np.array("barrier_handle_2", dtype=object),
        'keys': np.array(["matrix1", "matrix2"], dtype=object),
        'values': np.array([[-1, -2], [1, 2]], dtype=np.int32),
        'component_index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single key with a 3D tensor value
    input_dict_3 = {
        'name': None,
        'handle': np.array("barrier_handle_3", dtype=object),
        'keys': np.array(["tensor_3d"], dtype=object),
        'values': np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Empty keys and values
    input_dict_4 = {
        'name': 'insert_empty',
        'handle': np.array("barrier_handle_4", dtype=object),
        'keys': np.array([], dtype=object),
        'values': np.zeros(shape=(0, 10), dtype=np.float32),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: int64 values
    input_dict_5 = {
        'name': 'insert_int64',
        'handle': np.array("barrier_handle_5", dtype=object),
        'keys': np.array(["large_int"], dtype=object),
        'values': np.array([123456789012345], dtype=np.int64),
        'component_index': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: float64 values with negative numbers
    input_dict_6 = {
        'name': None,
        'handle': np.array("barrier_handle_6", dtype=object),
        'keys': np.array(["f64_1", "f64_2", "f64_3", "f64_4"], dtype=object),
        'values': np.array([1.0, -2.5e-10, 3e20, -np.pi], dtype=np.float64),
        'component_index': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: String values
    input_dict_7 = {
        'name': 'insert_strings',
        'handle': np.array("barrier_handle_7", dtype=object),
        'keys': np.array(["msg1", "msg2"], dtype=object),
        'values': np.array(["hello barrier", "world"], dtype=object),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Boolean values in a 2D tensor
    input_dict_8 = {
        'name': 'insert_booleans',
        'handle': np.array("barrier_handle_8", dtype=object),
        'keys': np.array(["flag_a", "flag_b", "flag_c"], dtype=object),
        'values': np.array([[True], [False], [True]], dtype=bool),
        'component_index': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large number of keys
    num_keys = 10
    input_dict_9 = {
        'name': 'insert_many_keys',
        'handle': np.array("barrier_handle_9", dtype=object),
        'keys': np.array([f"item_{i}" for i in range(num_keys)], dtype=object),
        'values': np.arange(num_keys * 2, dtype=np.int32).reshape(num_keys, 2),
        'component_index': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High component_index
    input_dict_10 = {
        'name': 'high_component_index',
        'handle': np.array("barrier_handle_10", dtype=object),
        'keys': np.array(["k1"], dtype=object),
        'values': np.array([[255, 0]], dtype=np.uint8),
        'component_index': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.BarrierInsertMany"] = tf_raw_ops_barrier_insert_many_inputs()

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
