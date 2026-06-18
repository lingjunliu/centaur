
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Cumsum_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_1d_float",
        'x': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        'axis': np.array(0, dtype=np.int32)
    })

    # Input 2
    list_of_inputs.append({
        'exclusive': True,
        'reverse': False,
        'name': "cumsum_1d_int_exclusive",
        'x': np.array([5, 10, 15], dtype=np.int32),
        'axis': np.array(0, dtype=np.int32)
    })

    # Input 3
    list_of_inputs.append({
        'exclusive': False,
        'reverse': True,
        'name': "cumsum_2d_double_reverse",
        'x': np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
        'axis': np.array(1, dtype=np.int32)
    })

    # Input 4
    list_of_inputs.append({
        'exclusive': True,
        'reverse': True,
        'name': "cumsum_2d_int64_both",
        'x': np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64),
        'axis': np.array(0, dtype=np.int64)
    })

    # Input 5
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_3d_float",
        'x': np.array([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32),
        'axis': np.array(2, dtype=np.int32)
    })

    # Input 6
    list_of_inputs.append({
        'exclusive': True,
        'reverse': False,
        'name': "cumsum_3d_int_axis_neg",
        'x': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        'axis': np.array(-1, dtype=np.int32)
    })

    # Input 7
    list_of_inputs.append({
        'exclusive': False,
        'reverse': True,
        'name': "cumsum_4d_float",
        'x': np.ones((2, 2, 2, 2), dtype=np.float32),
        'axis': np.array(1, dtype=np.int32)
    })

    # Input 8
    list_of_inputs.append({
        'exclusive': True,
        'reverse': True,
        'name': "cumsum_complex",
        'x': np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),
        'axis': np.array(0, dtype=np.int32)
    })

    # Input 9
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_int32_2",
        'x': np.array([[10, 20], [30, 40]], dtype=np.int32),
        'axis': np.array(-2, dtype=np.int64)
    })

    # Input 10
    list_of_inputs.append({
        'exclusive': False,
        'reverse': False,
        'name': "cumsum_double_1d",
        'x': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        'axis': np.array(0, dtype=np.int32)
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Cumsum"] = tf_raw_ops_Cumsum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cumsum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cumsum', generated_inputs['tf.raw_ops.Cumsum'], lib="tf", suffix=0)
