
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_insert_many_inputs():
    list_of_inputs = []

    # The 'handle' and 'keys' parameters require a 'string' type. When creating these
    # with numpy, a fixed-length string dtype (e.g., 'S20') is often inferred,
    # which can cause validation errors. To avoid this, we explicitly use
    # `dtype=object` for string tensors, creating an array of Python `bytes`
    # objects, which is correctly interpreted.
    dummy_handle_np = np.array(b"dummy_barrier_handle", dtype=object)

    # Input 1: Basic case, float32 scalars
    input_dict_1 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k1_a", b"k1_b", b"k1_c"], dtype=object),
        'values': np.array([10.1, 20.2, 30.3], dtype=np.float32),
        'component_index': 0,
        'name': 'insert_float_scalars'
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: int32 vectors with negative values
    input_dict_2 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k2_a", b"k2_b"], dtype=object),
        'values': np.array([[1, 2, 3], [-4, -5, -6]], dtype=np.int32),
        'component_index': 1,
        'name': 'insert_int_vectors'
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: complex128 matrices, single key
    input_dict_3 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k3_a"], dtype=object),
        'values': np.array([[[1+2j, 3-4j], [5+6j, 7-8j]]], dtype=np.complex128),
        'component_index': 0,
        'name': 'insert_complex_matrix'
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: string scalars, no name
    input_dict_4 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k4_a", b"k4_b", b"k4_c", b"k4_d"], dtype=object),
        'values': np.array([b"val_a", b"val_b", b"val_c", b"val_d"], dtype=object),
        'component_index': 1,
        'name': None
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: High-dimensional int64 tensor
    input_dict_5 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k5_a", b"k5_b"], dtype=object),
        'values': np.array([[[[1, 2, 3]], [[4, 5, 6]]], [[[7, 8, 9]], [[10, 11, 12]]]], dtype=np.int64),
        'component_index': 0,
        'name': 'insert_int64_3d'
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Empty keys and values (valid no-op)
    input_dict_6 = {
        'handle': dummy_handle_np,
        'keys': np.array([], dtype=object),
        'values': np.empty((0,), dtype=np.float32),
        'component_index': 0,
        'name': 'insert_empty'
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Empty keys/values for a multi-dimensional component
    input_dict_7 = {
        'handle': dummy_handle_np,
        'keys': np.array([], dtype=object),
        'values': np.empty((0, 2, 1, 3), dtype=np.int64),
        'component_index': 0,
        'name': 'insert_empty_multidim'
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Single uint8 value
    input_dict_8 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k8_a"], dtype=object),
        'values': np.array([[100, 200]], dtype=np.uint8),
        'component_index': 0,
        'name': 'insert_single_uint8'
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Boolean values
    input_dict_9 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k9_a", b"k9_b"], dtype=object),
        'values': np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=np.bool_),
        'component_index': 0,
        'name': 'insert_booleans'
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: float16 (half precision) values
    input_dict_10 = {
        'handle': dummy_handle_np,
        'keys': np.array([b"k10_a", b"k10_b"], dtype=object),
        'values': np.array([0.5, 1.5], dtype=np.float16),
        'component_index': 0,
        'name': 'insert_float16'
    }
    list_of_inputs.append(input_dict_10)

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
