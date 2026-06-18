
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []

    # Input 1: 1-D float32, axis=[0]
    list_of_inputs.append({
        "tensor": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "axis": np.array([0], dtype=np.int32),
        "name": "reverse_1d_float32"
    })

    # Input 2: 1-D int32, axis=[-1]
    list_of_inputs.append({
        "tensor": np.array([10, 20, 30], dtype=np.int32),
        "axis": np.array([-1], dtype=np.int32),
        "name": "reverse_1d_int32"
    })

    # Input 3: 2-D bool, axis=[0]
    list_of_inputs.append({
        "tensor": np.array([[True, False], [False, True]], dtype=np.bool_),
        "axis": np.array([0], dtype=np.int32),
        "name": "reverse_2d_bool"
    })

    # Input 4: 2-D uint8, axis=[1]
    list_of_inputs.append({
        "tensor": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        "axis": np.array([1], dtype=np.int32),
        "name": "reverse_2d_uint8"
    })

    # Input 5: 2-D float64, axis=[0, 1]
    list_of_inputs.append({
        "tensor": np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64),
        "axis": np.array([0, 1], dtype=np.int32),
        "name": "reverse_2d_multi"
    })

    # Input 6: 3-D int64, axis=[0, 2]
    list_of_inputs.append({
        "tensor": np.arange(8, dtype=np.int64).reshape((2, 2, 2)),
        "axis": np.array([0, 2], dtype=np.int64),
        "name": "reverse_3d_int64"
    })

    # Input 7: 3-D float32, axis=[-1, -3]
    list_of_inputs.append({
        "tensor": np.arange(24, dtype=np.float32).reshape((2, 3, 4)),
        "axis": np.array([-1, -3], dtype=np.int32),
        "name": "reverse_3d_neg"
    })

    # Input 8: 4-D complex64, axis=[1, 3]
    list_of_inputs.append({
        "tensor": np.arange(16, dtype=np.complex64).reshape((2, 2, 2, 2)),
        "axis": np.array([1, 3], dtype=np.int32),
        "name": "reverse_4d_complex"
    })

    # Input 9: 5-D string tensor, axis=[4]
    list_of_inputs.append({
        "tensor": np.array([[[[["a", "b"], ["c", "d"]]]]], dtype=np.object_),
        "axis": np.array([4], dtype=np.int32),
        "name": "reverse_5d_string"
    })

    # Input 10: 8-D int16, axis=[7]
    list_of_inputs.append({
        "tensor": np.arange(256, dtype=np.int16).reshape((2, 2, 2, 2, 2, 2, 2, 2)),
        "axis": np.array([7], dtype=np.int32),
        "name": "reverse_8d_max"
    })

    # Input 11: 2-D with empty axis
    list_of_inputs.append({
        "tensor": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "axis": np.array([], dtype=np.int32),
        "name": "reverse_empty_axis"
    })

    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse'], lib="tf", suffix=0)
