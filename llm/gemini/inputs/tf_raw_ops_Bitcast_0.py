
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Bitcast_inputs():
    list_of_inputs = []

    # Input 1: Equal size, float32 to int32 conversion (1D array)
    list_of_inputs.append({
        'name': 'eq_size_i32_to_f32',
        'input': np.array([1, 2, -3], dtype=np.int32),
        'type': np.dtype('float32')
    })

    # Input 2: Larger to smaller, float32 (4 bytes) to uint8 (1 byte)
    list_of_inputs.append({
        'name': 'f32_to_u8',
        'input': np.array([1.0, -2.0], dtype=np.float32),
        'type': np.dtype('uint8')
    })

    # Input 3: Smaller to larger, uint8 (1 byte) to int32 (4 bytes)
    list_of_inputs.append({
        'name': 'u8_to_i32',
        'input': np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8),
        'type': np.dtype('int32')
    })

    # Input 4: Equal size, complex64 (8 bytes) to int64 (8 bytes)
    list_of_inputs.append({
        'name': 'c64_to_i64',
        'input': np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64),
        'type': np.dtype('int64')
    })

    # Input 5: Equal size, int64 to float64 (1D array)
    list_of_inputs.append({
        'name': 'i64_to_f64',
        'input': np.array([1, -2, 3], dtype=np.int64),
        'type': np.dtype('float64')
    })

    # Input 6: Larger to smaller, float64 (8 bytes) to float32 (4 bytes) in 2D
    list_of_inputs.append({
        'name': 'f64_to_f32',
        'input': np.array([[1.0], [2.0]], dtype=np.float64),
        'type': np.dtype('float32')
    })

    # Input 7: Smaller to larger, float32 (4 bytes) to float64 (8 bytes)
    list_of_inputs.append({
        'name': 'f32_to_f64',
        'input': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        'type': np.dtype('float64')
    })

    # Input 8: Equal size, negative values, int16 to uint16
    list_of_inputs.append({
        'name': 'i16_to_u16',
        'input': np.array([-10, 0, 10], dtype=np.int16),
        'type': np.dtype('uint16')
    })

    # Input 9: Larger to smaller, int32 (4 bytes) to int16 (2 bytes)
    list_of_inputs.append({
        'name': 'i32_to_i16',
        'input': np.array([123456], dtype=np.int32),
        'type': np.dtype('int16')
    })

    # Input 10: Smaller to larger, int16 (2 bytes) to int32 (4 bytes)
    list_of_inputs.append({
        'name': 'i16_to_i32',
        'input': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16),
        'type': np.dtype('int32')
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Bitcast"] = tf_raw_ops_Bitcast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bitcast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bitcast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bitcast', generated_inputs['tf.raw_ops.Bitcast'], lib="tf", suffix=0)
